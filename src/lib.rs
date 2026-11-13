//! Tard: AlphaZero-style self-play training for two-player board games.
//!
//! - `device`: picking the device (CUDA if usable, else the CPU)
//! - `games`: the game interface and each game's rules
//! - `mcts`: Monte Carlo tree search guided by the network
//! - `model`: the policy/value network and model files
//! - `evaluator`: running the network for MCTS, alone or shared by threads
//! - `selfplay`: playing self-play games, on one thread or a worker pool
//! - `training`: replay buffer, loss, and the run directory
//!
//! The `tard` binary (src/main.rs, src/commands/) is the command line on top.
pub mod device;
pub mod evaluator;
pub mod games;
pub mod mcts;
pub mod model;
pub mod selfplay;
pub mod training;
pub mod util;
