//! `tard train`: trains a model by self-play.
//!
//! Each iteration:
//!   1) collect self-play games from the SelfPlaySource for --mode
//!   2) run --steps-per-iter training steps on samples from the replay buffer
//!   3) push the new weights back into the source, save a checkpoint, and
//!      append a row to metrics.csv
//!
//! Self-play modes (see src/selfplay.rs and src/evaluator/):
//!   single     - this thread plays games inline with a local model.
//!   batched    - worker threads send eval requests to one server thread
//!                that batches them on the GPU.
//!   replicated - each worker thread keeps its own model copy, refreshed
//!                whenever new weights are published.
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{Device, Tensor};
use clap::ValueEnum;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};

use tard::evaluator::{BatchedBackend, ReplicatedBackend};
use tard::games::{Game, GameState};
use tard::mcts::DirichletHParams;
use tard::model::{load_model, save_model, NetConfig, Tard};
use tard::selfplay::{SelfPlaySource, SingleThreaded, WorkerPool};
use tard::training::{
    compute_loss, selfplay_to_training, step_from_checkpoint_dir, IterationMetrics, OutputManager,
    TrainingBuffer,
};
use tard::with_game;

#[derive(Copy, Clone, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Single,
    Batched,
    Replicated,
}

// Serialized as the run's config.json.
#[derive(clap::Args, Debug, Serialize)]
pub struct Args {
    /// Which game to train
    #[arg(long, value_enum, default_value_t = Game::MiniPente)]
    game: Game,

    /// How self-play games are generated
    #[arg(long, value_enum, default_value_t = Mode::Batched)]
    mode: Mode,

    /// Number of self-play worker threads (ignored in single mode)
    #[arg(long, default_value_t = 200)]
    workers: usize,

    /// MCTS visits per move. Self-play keeps the chosen move's subtree, and
    /// its visits count toward this
    #[arg(long, default_value_t = 800)]
    rollouts: usize,

    /// Self-play samples moves proportionally to MCTS visits for this many
    /// plies, then plays the most-visited move
    #[arg(long, default_value_t = 10)]
    temp_moves: usize,

    /// Max batch size for the batched eval server
    #[arg(long, default_value_t = 1024)]
    eval_batch_size: usize,

    /// Checkpoint directory to continue a run from
    /// (`<run>/checkpoints/step_<N>`): restores the weights, the optimizer
    /// state, and step N. Without it, training starts from a fresh network.
    #[arg(long)]
    resume: Option<PathBuf>,

    // ---- Network size (only for a freshly initialized network) ----
    // A resumed model keeps the size saved with it; passing a value that
    // disagrees with it is an error.
    /// Channels in the conv trunk [default: the game's default: 64 for MiniPente, 128 for Pente, 64 for Hive]
    #[arg(long)]
    channels: Option<usize>,

    /// Residual blocks in the trunk [default: the game's default: 6 for MiniPente, 10 for Pente, 8 for Hive]
    #[arg(long)]
    blocks: Option<usize>,

    /// Hidden units in the value head [default: the game's default: 64 for MiniPente, 128 for Pente, 64 for Hive]
    #[arg(long)]
    value_hidden: Option<usize>,

    /// Root directory for outputs. Each run writes to a new
    /// `<output-dir>/run_<unix_seconds>/` subdirectory.
    #[arg(long, default_value = "output")]
    output_dir: PathBuf,

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

impl Args {
    // `base` with any network-size flags applied on top.
    fn net_config(&self, base: NetConfig) -> NetConfig {
        return NetConfig {
            channels: self.channels.unwrap_or(base.channels),
            blocks: self.blocks.unwrap_or(base.blocks),
            value_hidden: self.value_hidden.unwrap_or(base.value_hidden),
        };
    }
}

pub fn run(args: Args) {
    with_game!(args.game, G => train_tard::<G>(&args, args.game.default_net()));
}

// Trains a model for game G. `default_net` sizes a fresh network unless
// overridden by --channels/--blocks/--value-hidden.
fn train_tard<G: GameState>(args: &Args, default_net: NetConfig) {
    info!("Game: {}", G::NAME);
    ///// Configs ////////////
    let n_training_steps_per_iter = args.steps_per_iter;
    let batch_size = args.batch_size;
    let n_elements_per_train = n_training_steps_per_iter * batch_size / args.sample_reuse.max(1);
    let first_n_elements_train = args.warmup_iters * n_elements_per_train;
    let train_mem_capacity: usize = args.buffer_iters * n_elements_per_train;
    let dirchlet_hyper_params = DirichletHParams {
        epsilon: args.dirichlet_epsilon,
        alpha: args
            .dirichlet_alpha
            .unwrap_or(20.0 / G::ACTN_SPC_SIZE as f32),
    };
    let max_learning_rate = args.max_lr;
    let min_learning_rate = args.min_lr;
    let n_training_steps = args.total_steps;

    let start_time = Instant::now();
    let device = Device::default().autodiff();
    info!("Using Device: {:?}", device);
    let mut opt = AdamWConfig::new()
        .with_weight_decay(args.weight_decay)
        .init();
    let mut lr = CosineAnnealingLrSchedulerConfig::new(max_learning_rate, n_training_steps)
        .with_min_lr(min_learning_rate)
        .init()
        .unwrap();
    let mut train_step_cnt = 0;
    let (mut tard, net) = if let Some(ckpt) = &args.resume {
        train_step_cnt = step_from_checkpoint_dir(ckpt).unwrap_or_else(|| {
            panic!(
                "--resume {} is not a checkpoint directory named step_<N>",
                ckpt.display()
            )
        });
        info!(
            "Resuming from checkpoint {} at step {}/{}",
            ckpt.display(),
            train_step_cnt,
            n_training_steps
        );
        let model_path = ckpt.join("model.bpk");
        let loaded = load_model::<G>(&model_path, &device)
            .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model_path.display(), e));
        let optim_path = ckpt.join("optim.bpk");
        opt = opt.load(&optim_path).unwrap_or_else(|e| {
            panic!("Failed to load optimizer {}: {:?}", optim_path.display(), e)
        });
        // Fast-forward the LR schedule to the resumed step.
        for _ in 0..train_step_cnt {
            lr.step();
        }
        info!("Replay buffer is not persisted and will refill from self-play.");
        loaded
    } else {
        info!("No --resume given; creating a fresh network");
        let net = args.net_config(default_net);
        (Tard::for_game::<G>(&net, &device), net)
    };
    // A resumed model's saved size wins; flags may only restate it.
    let requested = args.net_config(net);
    if requested != net {
        panic!(
            "Network-size flags ask for {:?}, but the loaded model is {:?}. \
                --channels/--blocks/--value-hidden only apply to a fresh network.",
            requested, net
        );
    }
    info!("Network: {:?}", net);
    debug!("{}", tard);
    // Created only after loading succeeds, so a bad --resume doesn't leave
    // an empty run directory behind.
    let config_json = serde_json::to_string_pretty(args).expect("Unable to serialize config");
    let mut om = OutputManager::new(&args.output_dir, &config_json).unwrap();
    let mut trn_buf = TrainingBuffer::new(train_mem_capacity);

    info!("Self-play mode: {:?}", args.mode);
    let mut source: Box<dyn SelfPlaySource<G>> = match args.mode {
        Mode::Single => Box::new(SingleThreaded::new(
            &tard,
            args.rollouts,
            dirchlet_hyper_params,
            args.temp_moves,
        )),
        Mode::Batched => Box::new(WorkerPool::new(
            BatchedBackend::new(&tard, args.eval_batch_size),
            args.workers,
            args.rollouts,
            dirchlet_hyper_params,
            args.temp_moves,
        )),
        Mode::Replicated => Box::new(WorkerPool::new(
            ReplicatedBackend::new(&tard),
            args.workers,
            args.rollouts,
            dirchlet_hyper_params,
            args.temp_moves,
        )),
    };

    let mut is_running = true;
    let mut el_cnt_threshold = first_n_elements_train;
    let mut current_lr = ModuleLearningRate::default();
    while is_running && train_step_cnt < n_training_steps {
        let iter_start = Instant::now();
        let evals_at_start = source.eval_stats();
        info!("Awaiting Self-Play Games");
        let mut el_cnt = 0;
        let (mut games, mut draws) = (0, 0);
        while el_cnt < el_cnt_threshold {
            match source.next_game() {
                Some(game) => {
                    el_cnt += game.len();
                    games += 1;
                    if game.last().is_some_and(|e| e.outcome == 0.0) {
                        draws += 1;
                    }
                    let train_vec = selfplay_to_training(game);
                    trn_buf.extend(train_vec);
                }
                None => {
                    warn!(
                        "Self-play source closed unexpectedly... signaling shutdown of main loop"
                    );
                    is_running = false;
                    break;
                }
            }
            info!(
                "Read {}/{} game elements for next training loop",
                el_cnt, el_cnt_threshold
            );
        }
        if !is_running {
            break;
        }
        el_cnt_threshold = n_elements_per_train; // After epoch 0, use this instead
        let selfplay_wait = iter_start.elapsed();

        info!(
            "Performing training {}/{}",
            train_step_cnt, n_training_steps
        );
        let train_start = Instant::now();
        // Per-step losses are summed on the device (detached from the graph)
        // and read back once, so logging doesn't force a GPU sync every step.
        let mut sum_loss = Tensor::<1>::zeros([1], &device);
        let mut sum_ce = Tensor::<1>::zeros([1], &device);
        let mut sum_mse = Tensor::<1>::zeros([1], &device);
        let mut n_steps = 0;
        for _ in 0..n_training_steps_per_iter {
            // Stop at the end of the cosine schedule; past n_training_steps
            // burn's cosine LR rises back toward max_learning_rate.
            if train_step_cnt >= n_training_steps {
                break;
            }
            let trn_samples = trn_buf.sample(batch_size);
            let t_loss = compute_loss::<G>(&tard, &trn_samples);
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
        let train_time = train_start.elapsed(); // after the readback, so GPU work is included
        info!(
            "Training batch completed. Progress {}/{} complete",
            train_step_cnt, n_training_steps
        );
        info!(
            "Mean loss over {} steps: {} (policy: {}, value: {})",
            n_steps, mean_loss, mean_ce, mean_mse
        );

        info!("Updating self-play model");
        source.update_model(&tard);

        let ckpt = om
            .checkpoint_dir(train_step_cnt)
            .expect("Unable to create checkpoint directory");
        info!("Saving checkpoint {}", ckpt.display());
        save_model::<G>(&tard, &net, &ckpt.join("model.bpk"))
            .unwrap_or_else(|e| panic!("Unable to save model: {}", e));
        opt.save(ckpt.join("optim.bpk"))
            .expect("Unable to save optimizer state");
        let iter_time = iter_start.elapsed();
        let evals = source.eval_stats().since(&evals_at_start);
        info!(
            "Iteration took {:.1}s (self-play wait {:.1}s, training {:.1}s): {:.1} positions/s, \
               {:.0} evals/s, mean eval batch {:.1}, eval busy {:.0}%",
            iter_time.as_secs_f64(),
            selfplay_wait.as_secs_f64(),
            train_time.as_secs_f64(),
            el_cnt as f64 / iter_time.as_secs_f64(),
            evals.positions as f64 / iter_time.as_secs_f64(),
            evals.positions as f64 / evals.batches.max(1) as f64,
            100.0 * evals.busy.as_secs_f64() / iter_time.as_secs_f64()
        );
        om.append_metrics(&IterationMetrics {
            step: train_step_cnt,
            wall_time_s: start_time.elapsed().as_secs_f64(),
            lr: current_lr.base(),
            loss: mean_loss,
            policy_loss: mean_ce,
            value_loss: mean_mse,
            games,
            positions: el_cnt,
            draws,
            iter_time_s: iter_time.as_secs_f64(),
            selfplay_wait_s: selfplay_wait.as_secs_f64(),
            train_s: train_time.as_secs_f64(),
            evals: evals.positions,
            eval_batches: evals.batches,
            eval_busy_s: evals.busy.as_secs_f64(),
        })
        .expect("Unable to write metrics");
    }
    info!(
        "Main thread loop over at training step {}/{}",
        train_step_cnt, n_training_steps
    );
    drop(source); // joins worker threads, if any
    info!("Main thread shutting down");
}
