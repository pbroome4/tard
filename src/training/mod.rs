//! What the training loop (`tard train`, src/commands/train.rs) is built
//! from: the replay buffer of self-play positions, the loss, and the run's
//! output directory.
pub mod replay;
pub mod loss;
pub mod run_dir;

pub use replay::{selfplay_to_training, TrainingBuffer, TrainingElement};
pub use loss::{compute_loss, TardLoss, TrainingBatch};
pub use run_dir::{step_from_checkpoint_dir, IterationMetrics, OutputManager};
