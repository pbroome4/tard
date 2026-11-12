use std::fs::{OpenOptions};
use std::io::{Write};
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn::module::Module;
use clap::{Parser, ValueEnum};
use tracing::{debug, info, warn};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use burn::tensor::{Device, Tensor};
use burn::optim::{AdamWConfig, GradientsParams};
use std::path::{Path, PathBuf};

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

    /// Self-play samples moves proportionally to MCTS visits for this many
    /// plies, then plays the most-visited move
    #[arg(long, default_value_t = 10)]
    temp_moves: usize,

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

    /// Training step to resume from. Defaults to <N> parsed from a
    /// `model_<N>.bpk` --model filename, else 0.
    #[arg(long)]
    start_step: Option<usize>,

    // ---- Training hyperparameters ----

    /// Total training steps; the cosine LR schedule spans exactly this many
    #[arg(long, default_value_t = 200_000)]
    total_steps: usize,

    /// Training steps between self-play model updates / checkpoints
    #[arg(long, default_value_t = 512)]
    steps_per_iter: usize,

    /// Positions per training batch
    #[arg(long, default_value_t = 128)]
    batch_size: usize,

    /// Average number of times each new position is sampled for training.
    /// Each iteration waits for steps_per_iter * batch_size / sample_reuse
    /// new positions.
    #[arg(long, default_value_t = 2)]
    sample_reuse: usize,

    /// The first iteration waits for this many iterations' worth of positions
    #[arg(long, default_value_t = 5)]
    warmup_iters: usize,

    /// Replay buffer holds this many iterations' worth of positions
    #[arg(long, default_value_t = 10)]
    buffer_iters: usize,

    /// Peak (initial) learning rate of the cosine schedule
    #[arg(long, default_value_t = 0.01)]
    max_lr: f64,

    /// Final learning rate of the cosine schedule
    #[arg(long, default_value_t = 0.0002)]
    min_lr: f64,

    /// AdamW weight decay
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f32,

    /// Fraction of the root prior replaced by Dirichlet noise in self-play
    #[arg(long, default_value_t = 0.2)]
    dirichlet_epsilon: f32,

    /// Dirichlet concentration. Defaults to 20 / action-space size
    #[arg(long)]
    dirichlet_alpha: Option<f32>,
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
    let n_training_steps_per_iter = args.steps_per_iter;
    let batch_size = args.batch_size;
    let n_elements_per_train = n_training_steps_per_iter * batch_size / args.sample_reuse.max(1);
    let first_n_elements_train = args.warmup_iters * n_elements_per_train;
    let train_mem_capacity: usize = args.buffer_iters * n_elements_per_train;
    let dirchlet_hyper_params = DirichletHParams{
        epsilon: args.dirichlet_epsilon,
        alpha: args.dirichlet_alpha.unwrap_or(20.0 / PenteState::ACTN_SPC_SIZE as f32),
    };
    let max_learning_rate = args.max_lr;
    let min_learning_rate = args.min_lr;
    let n_training_steps = args.total_steps;

    let om = OutputManager::new(&args.output_dir).unwrap();
    let device = Device::default().autodiff();
    info!("Using Device: {:?}", device);
    let mut tard = if args.model.exists(){
        info!("Resuming from model {}", args.model.display());
        load_model::<PenteState>(&args.model, &device).unwrap_or_else(|e|{
            panic!("Failed to load model {}: {:?}", args.model.display(), e)
        })
    }else{
        info!("No model at {}, creating fresh instance", args.model.display());
        Tard::for_game::<PenteState>(&device)
    };
    debug!("{}", tard);
    let mut opt = AdamWConfig::new().with_weight_decay(args.weight_decay).init();
    let mut lr = CosineAnnealingLrSchedulerConfig::new(max_learning_rate, n_training_steps)
        .with_min_lr(min_learning_rate)
        .init()
        .unwrap();
    let mut train_step_cnt = 0;
    if args.model.exists(){
        train_step_cnt = args.start_step
            .or_else(|| step_from_model_path(&args.model))
            .unwrap_or_else(||{
                warn!("Could not determine training step for {}; starting at step 0 (pass --start-step to override)",
                      args.model.display());
                0
            });
        let optim_path = args.model.with_file_name(format!("optim_{}.bpk", train_step_cnt));
        if optim_path.exists(){
            info!("Resuming optimizer state from {}", optim_path.display());
            opt = opt.load(&optim_path).unwrap_or_else(|e|{
                panic!("Failed to load optimizer {}: {:?}", optim_path.display(), e)
            });
        }else{
            warn!("No optimizer state at {}; using a fresh optimizer", optim_path.display());
        }
        // Fast-forward the LR schedule to the resumed step.
        for _ in 0..train_step_cnt{
            lr.step();
        }
        info!("Resuming at training step {}/{}. Replay buffer is not persisted and will refill from self-play.",
              train_step_cnt, n_training_steps);
    }
    let mut trn_buf = tard::TrainingBuffer::new(train_mem_capacity);

    info!("Self-play mode: {:?}", args.mode);
    let mut source: Box<dyn SelfPlaySource<PenteState>> = match args.mode{
        Mode::Single => Box::new(SingleThreaded::new(
            &tard, args.rollouts, dirchlet_hyper_params, args.temp_moves)),
        Mode::Batched => Box::new(WorkerPool::new(
            BatchedBackend::new(&tard, args.eval_batch_size),
            args.workers, args.rollouts, dirchlet_hyper_params, args.temp_moves)),
        Mode::Replicated => Box::new(WorkerPool::new(
            ReplicatedBackend::new(&tard),
            args.workers, args.rollouts, dirchlet_hyper_params, args.temp_moves)),
    };

    let mut is_running = true;
    let mut el_cnt_threshold = first_n_elements_train;
    let mut current_lr = ModuleLearningRate::default();
    while is_running && train_step_cnt < n_training_steps{
        info!("Awaiting Self-Play Games");
        let mut el_cnt = 0;
        while el_cnt < el_cnt_threshold{
            match source.next_game(){
                Some(game) =>{
                    el_cnt += game.len();
                    let train_vec = selfplay_to_training(game);
                    trn_buf.extend(train_vec);
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
        // Per-step losses are summed on the device (detached from the graph)
        // and read back once, so logging doesn't force a GPU sync every step.
        let mut sum_loss = Tensor::<1>::zeros([1], &device);
        let mut sum_ce = Tensor::<1>::zeros([1], &device);
        let mut sum_mse = Tensor::<1>::zeros([1], &device);
        let mut n_steps = 0;
        for _ in 0..n_training_steps_per_iter{
            // Stop at the end of the cosine schedule; past n_training_steps
            // burn's cosine LR rises back toward max_learning_rate.
            if train_step_cnt >= n_training_steps{
                break;
            }
            let trn_samples = trn_buf.sample(batch_size);
            let t_loss = compute_loss::<PenteState>(&tard, &trn_samples);
            debug!("t_loss: {}", t_loss);
            sum_loss = sum_loss + t_loss.loss.clone().detach();
            sum_ce = sum_ce + t_loss.cross_entropy_loss.clone().detach();
            sum_mse = sum_mse + t_loss.mse_loss.clone().detach();
            n_steps += 1;
            let grads = t_loss.loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &tard);
            current_lr = lr.step();
            tard = opt.step(current_lr.base(), tard, grad_params);
            train_step_cnt += 1;
        }
        let mean = |t: Tensor<1>| t.into_scalar::<f32>() / n_steps.max(1) as f32;
        let (mean_loss, mean_ce, mean_mse) = (mean(sum_loss), mean(sum_ce), mean(sum_mse));
        info!("Training batch completed. Progress {}/{} complete", train_step_cnt, n_training_steps);
        info!("Mean loss over {} steps: {} (cross_entropy: {}, mse: {})",
              n_steps, mean_loss, mean_ce, mean_mse);

        info!("Updating self-play model");
        source.update_model(&tard);

        info!("Saving model");
        let model_name = format!("model_{}", train_step_cnt);
        tard.clone().save_file(om.model_d.join(model_name))
                    .expect("Unable to save model");
        opt.save(om.model_d.join(format!("optim_{}.bpk", train_step_cnt)))
           .expect("Unable to save optimizer state");
        info!("Saving loss");
        let mut lf = OpenOptions::new()
                                .write(true)
                                .append(true)
                                .open(&om.loss_f).unwrap();
        writeln!(lf, "{}, {}, {}, {}", mean_loss, mean_ce, mean_mse, current_lr.base()).unwrap();
    }
    info!("Main thread loop over at training step {}/{}", train_step_cnt, n_training_steps);
    drop(source); // joins worker threads, if any
    info!("Main thread shutting down");
}

// Parses <N> from a checkpoint path of the form `.../model_<N>.bpk`.
fn step_from_model_path(p: &Path) -> Option<usize>{
    p.file_stem()?
     .to_str()?
     .strip_prefix("model_")?
     .parse()
     .ok()
}
