use std::fs::{OpenOptions};
use std::io::{Write};
use std::thread::{self, JoinHandle};
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn::module::Module;
use tracing::{debug, info, warn};
use tracing_subscriber::{prelude::*};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use burn::tensor::{Device};
use burn::optim::{AdamConfig, AdamWConfig, GradientsParams};
use crossbeam::channel::{Receiver, RecvError, SendError, Sender};
use std::path::{Path};

mod tard;
mod gamestate;
mod evaluator;
mod engine;
mod pente;
mod utils;

use crate::gamestate::GameState;
use crate::pente::PenteState;
use crate::engine::*;
use crate::evaluator::*;
use crate::utils::*;
use crate::tard::*;
////////////////////////////////////////
//      ## Main thread
//  1) Initialize model, vars, logging, etc  
//  2) Repeatedly self-play and train model
//      2.1) Spawn eval and worker threads
//      2.2) Collect worker data
//      2.3) Perform training
//      2.4) send update to eval thread.
//  3) Eventually terminate
//
//       #### Worker thread (many threads) ###
//   1) Create game at starting position (start with batch size 1)
//   2) Run MCTS
//       2.1) For each game, Run N(=800, 1600 etc?) MCTS runouts
//            This MCTS requires network prior and value output for PUCT
//            which requires a eval thread interaction
//       2.2) At the end of N MCTS runouts, make best move found for the game. 
//       2.3) repeat until game ends
//       2.4) Return training data IE:
//            For each state in each game, return (state, win/loss/draw),
//            network value, actual game terminal value(win/loss).
//   3) Send training data to a replay buffer in main thread
//
//      #### Evaluator thread ####
//  1) Recieve network eval requests on a queue
//  2) Batch up requests and send to network
//  3) Get evaluations and redistribute.
//  4) check for model updates from main thread.


fn main(){
    let mut filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into());
    filter = filter.add_directive("pentard=info".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    train_tard();
}

fn train_tard(){
    ///// Configs ////////////
    let n_worker_threads = 200;
    let n_rollouts = 800;
    let n_training_steps_per_iter = 512; 
    let batch_size = 128;
    let n_elements_per_train = n_training_steps_per_iter * batch_size / 2; // ~= data reuse of 2
    let first_n_elements_train = 5*n_elements_per_train;   
    let train_mem_capacity: usize = 10 * n_elements_per_train; // Keep most recent 10 model training generation states
    let dirchlet_hyper_params = DirichletHParams{
        epsilon: 0.2,
        alpha: 20.0 / PenteState::ACTN_SPC_SIZE as f32
    };
    let max_learning_rate = 0.01;
    let min_learning_rate = 0.0002;
    let n_training_steps = 200_000;

    let om = OutputManager::new("output", false).unwrap();
    let device = Device::default().autodiff();
    info!("Using Device: {:?}", device);
    info!("Attempting to load existing model");
    let mut tard = load_model(Path::new("model.bpk"), &device).unwrap_or_else(|_|{
        info!("No Tard model found to load, creating fresh instance");
        tard::Tard::new(
            PenteState::N_CHAN,
            PenteState::ROWS,
            PenteState::COLS,
            PenteState::ACTN_SPC_SIZE,
            &device)
    });
    debug!("{}", tard);
    let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();
    let mut lr = CosineAnnealingLrSchedulerConfig::new(max_learning_rate, n_training_steps)
        .with_min_lr(min_learning_rate)
        .init()
        .unwrap();
    let mut trn_buf = tard::TrainingBuffer::new(train_mem_capacity);
    let eval_srv = CBatchedTardEvalServer::new(&tard);
    let (s_games, r_games) = crossbeam::channel::bounded(4096);
    let handles = spawn_workers::<PenteState>(
        n_worker_threads,
        s_games,
        &eval_srv,
        n_rollouts,
        dirchlet_hyper_params);       
    let mut is_running = true;
    let mut train_step_cnt = 0;
    let mut el_cnt_threshold = first_n_elements_train;
    let mut t_loss = TardLoss::default();
    let mut current_lr = ModuleLearningRate::default();
    while is_running{
        info!("Awaiting Self-Play Games");
        let mut el_cnt = 0;
        while el_cnt < el_cnt_threshold{ 
            match r_games.recv(){
                Ok(req) =>{
                    el_cnt += req.len();
                    let train_vec = selfplay_to_training(req);
                    trn_buf.extend_front(train_vec.into_iter());
                }
                Err(RecvError) => {
                    warn!("Game Receiver closed unexpectedly... signaling shutdown of main loop");
                    is_running = false;
                    break
                }
            }
            info!("Read {}/{} game elements for next training loop", el_cnt, el_cnt_threshold);
        }
        el_cnt_threshold = n_elements_per_train; // After epoch 0, use this instead
        
        info!("Performing training {}/{}", train_step_cnt, n_training_steps);
        for _ in 0..n_training_steps_per_iter{
            let trn_samples = trn_buf.sample(batch_size);
            t_loss = compute_loss(&tard, &trn_samples);
            debug!("t_loss: {}", t_loss);
            let grads = t_loss.loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &tard);
            current_lr = lr.step();
            tard = opt.step(current_lr.base(), tard, grad_params);
            train_step_cnt += 1;
        }
        info!("Training batch completed. Progress {}/{} complete", train_step_cnt, n_training_steps);
        info!("Loss after training: {}", t_loss);
    
        info!("Updating Evaluator Thread");
        eval_srv.update(&tard);

        info!("Saving model");
        let model_name = format!("model_{}", train_step_cnt);
        tard.clone().save_file(om.model_d.join(model_name))
                    .expect("Unable to save model");
        info!("Saving loss");
        let mut lf = OpenOptions::new()
                                .write(true)
                                .append(true)
                                .open(&om.loss_f).unwrap();
        writeln!(lf,
                    "{}, {}, {}, {}",
                    t_loss.loss.clone()
                            .into_scalar::<f32>(),
                    t_loss.cross_entropy_loss.clone()
                                            .into_scalar::<f32>(),
                    t_loss.mse_loss.clone()
                                .into_scalar::<f32>(),
                    current_lr.base()
                ).unwrap();
    }
    info!("Main thread loop over");
    drop(r_games);
    for h in handles.into_iter(){
        h.join();
    }
    info!("Main thread shutting down");
}


fn spawn_workers <G:GameState>(
    n_workers: usize,
    s: Sender<Vec<SelfPlayElement<G>>>,
    eval_srv: &impl TardServer<G>,
    n_rollouts: usize,
    dhp: DirichletHParams) -> Vec<thread::JoinHandle<()>>
{
    let mut v = Vec::new(); 
    for i in 0..n_workers{
        let s_clone = s.clone();
        let eval = eval_srv.get_client();
        let h = start_worker(i, s_clone, eval, n_rollouts, dhp.clone());
        v.push(h);
    }
    return v;
}


fn start_worker<G:GameState>(
    thread_id: usize,
    s: Sender<Vec<SelfPlayElement<G>>>,
    mut eval: impl Evaluator<G>,
    n_rollouts: usize,
    dhp: DirichletHParams) -> JoinHandle<()>
{    
    info!("Spawning worker thread #{}", thread_id);
    thread::spawn(move ||{
        let mut is_running = true;
        let mut self_play_count = 0;
        while is_running{
            let game_data = run_self_play::<G>(&mut eval, n_rollouts, &dhp);
            debug!("Worker thread #{} Self play game {} complete.", thread_id, self_play_count);
            debug!("Worker thread #{} adding {} training elements", thread_id, game_data.len());
            match s.send(game_data){
                Ok(_)=>{}
                Err(SendError(_))=>{
                    info!("Worker {} Failed to send game state. Signalling shutdown.", thread_id);
                    is_running = false;
                    break;
                }   
            }
            self_play_count += 1;   
        }
        debug!("Worker #{} Shutting down", thread_id);
    })
}
