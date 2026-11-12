use std::fs::{OpenOptions};
use std::io::{Write};
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn::module::Module;
use clap::{Parser, ValueEnum};
use tracing::{debug, info, warn};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use burn::tensor::{Device};
use burn::optim::{AdamWConfig, GradientsParams};
use std::path::PathBuf;

use pentard::gamestate::GameState;
use pentard::pente::PenteState;
use pentard::engine::*;
use pentard::evaluator::{BatchedBackend, ReplicatedBackend};
use pentard::selfplay::{SelfPlaySource, SingleThreaded, WorkerPool};
use pentard::utils::*;
use pentard::tard::{self, *};
////////////////////////////////////////
//      ## Main thread
//  1) Initialize model, vars, logging, etc
//  2) Repeatedly self-play and train model
//      2.1) Build a SelfPlaySource for the chosen --mode
//      2.2) Collect self-play games from it
//      2.3) Perform training
//      2.4) Push the updated model back into the source
//  3) Eventually terminate
//
//  Self-play modes (see src/selfplay.rs and src/evaluator/):
//    single     - main thread plays games inline with a local model.
//    batched    - worker threads send eval requests to one server thread
//                 that batches them on the GPU.
//    replicated - each worker thread keeps its own model copy, refreshed
//                 whenever the main thread publishes new weights.


#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode{
    Single,
    Batched,
    Replicated,
}

#[derive(Parser, Debug)]
#[command(about = "Train Tard via MCTS self-play")]
struct Args{
    /// How self-play games are generated
    #[arg(long, value_enum, default_value_t = Mode::Batched)]
    mode: Mode,

    /// Number of self-play worker threads (ignored in single mode)
    #[arg(long, default_value_t = 200)]
    workers: usize,

    /// MCTS rollouts per move
    #[arg(long, default_value_t = 800)]
    rollouts: usize,

    /// Max batch size for the batched eval server
    #[arg(long, default_value_t = 1024)]
    eval_batch_size: usize,

    /// Model to resume training from. If the file doesn't exist, training
    /// starts from a freshly initialized model.
    #[arg(long, default_value = "model.bpk")]
    model: PathBuf,

    /// Root directory for outputs. Each run writes to a new
    /// `<output-dir>/run_<unix_seconds>/` subdirectory.
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,
}


fn main(){
    let args = Args::parse();
    let mut filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into());
    filter = filter.add_directive("pentard=info".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
    train_tard(&args);
}

fn train_tard(args: &Args){
    ///// Configs ////////////
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

    let om = OutputManager::new(&args.output_dir).unwrap();
    let device = Device::default().autodiff();
    info!("Using Device: {:?}", device);
    let mut tard = if args.model.exists(){
        info!("Resuming from model {}", args.model.display());
        load_model(&args.model, &device).unwrap_or_else(|e|{
            panic!("Failed to load model {}: {:?}", args.model.display(), e)
        })
    }else{
        info!("No model at {}, creating fresh instance", args.model.display());
        tard::Tard::new(
            PenteState::N_CHAN,
            PenteState::ROWS,
            PenteState::COLS,
            PenteState::ACTN_SPC_SIZE,
            &device)
    };
    debug!("{}", tard);
    let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();
    let mut lr = CosineAnnealingLrSchedulerConfig::new(max_learning_rate, n_training_steps)
        .with_min_lr(min_learning_rate)
        .init()
        .unwrap();
    let mut trn_buf = tard::TrainingBuffer::new(train_mem_capacity);

    info!("Self-play mode: {:?}", args.mode);
    let mut source: Box<dyn SelfPlaySource<PenteState>> = match args.mode{
        Mode::Single => Box::new(SingleThreaded::new(
            &tard, args.rollouts, dirchlet_hyper_params)),
        Mode::Batched => Box::new(WorkerPool::new(
            BatchedBackend::new(&tard, args.eval_batch_size),
            args.workers, args.rollouts, dirchlet_hyper_params)),
        Mode::Replicated => Box::new(WorkerPool::new(
            ReplicatedBackend::new(&tard),
            args.workers, args.rollouts, dirchlet_hyper_params)),
    };

    let mut is_running = true;
    let mut train_step_cnt = 0;
    let mut el_cnt_threshold = first_n_elements_train;
    let mut t_loss = TardLoss::default();
    let mut current_lr = ModuleLearningRate::default();
    while is_running{
        info!("Awaiting Self-Play Games");
        let mut el_cnt = 0;
        while el_cnt < el_cnt_threshold{
            match source.next_game(){
                Some(game) =>{
                    el_cnt += game.len();
                    let train_vec = selfplay_to_training(game);
                    trn_buf.extend_front(train_vec.into_iter());
                }
                None => {
                    warn!("Self-play source closed unexpectedly... signaling shutdown of main loop");
                    is_running = false;
                    break
                }
            }
            info!("Read {}/{} game elements for next training loop", el_cnt, el_cnt_threshold);
        }
        if !is_running{
            break;
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

        info!("Updating self-play model");
        source.update_model(&tard);

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
    drop(source); // joins worker threads, if any
    info!("Main thread shutting down");
}
