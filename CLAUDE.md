# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Tard" is an AlphaZero-style AI for the board game Pente (9x9 board, capture rules), written in Rust using the `burn` deep learning framework (crate name `pentard`). It trains a policy/value ResNet via MCTS self-play.

## Commands

```bash
cargo build --release                      # build (use --release for anything performance-sensitive)
cargo run --release --bin pentard -- --mode batched   # self-play training (src/main.rs); --mode single|batched|replicated, also --workers, --rollouts, --eval-batch-size
cargo run --release --bin inference        # play against the model in ./model.bpk
cargo run --release --bin interactive      # human-vs-human Pente in the terminal (no AI)
```

- Environment (from `commands.txt`): `export CUDA_PATH=/usr/local/cuda`, `RUST_BACKTRACE=1`. burn is built with the `cuda`, `fusion`, and `flex` features; `Device::default()` picks the backend.
- Logging uses `tracing` with `EnvFilter`; defaults are `warn` globally and `pentard=info`. Override with e.g. `RUST_LOG=pentard=debug`.
- Profiling: `cargo flamegraph --dev --min-width 0.0 -b pentard` (requires `sudo sysctl -w kernel.perf_event_paranoid=-1` and `kernel.kptr_restrict=0`).
- There is no test suite. `src/bin/burn_test.rs` and `src/bin/crossbeam_test.rs` are scratch experiments, not tests.

## Architecture

The training loop lives in `src/main.rs` (`train_tard`). Training hyperparameters (batch size, LR schedule, Dirichlet noise) are hardcoded locals at the top of it; worker count, rollouts, and mode come from CLI args. Each iteration:
1. Pulls finished games from a `Box<dyn selfplay::SelfPlaySource<G>>` (`next_game`) into a `tard::TrainingBuffer` (bounded replay buffer).
2. Runs `n_training_steps_per_iter` AdamW steps with cosine-annealed LR.
3. Pushes new weights back with `source.update_model(&tard)`, saves `output/models/model_<step>.bpk`, and appends to `output/loss.csv`.

Self-play sources (`--mode`), in `src/selfplay.rs`:
- `SingleThreaded`: runs `engine::run_self_play` inline with a local `TardEvaluator`.
- `WorkerPool<G, B: EvalBackend<G>>`: N worker threads, each with an `Evaluator` from `backend.client()`, sending games over a crossbeam channel. `Drop` closes the channel and joins the workers. Backends live in `src/evaluator/`:
  - `BatchedBackend` (`batched.rs`): workers send requests to one server thread, which batches up to `--eval-batch-size` positions per GPU forward pass. Weight updates go to the server thread over a channel.
  - `ReplicatedBackend` (`replicated.rs`): each worker owns a model copy. `update` writes into a shared `Mutex<Tard>` and bumps an `AtomicU64` version; a client re-clones the model on its next `evaluate` after the version changes.
- Models handed to evaluators are always `tard.valid()` (the non-autodiff model).

Key abstractions:
- **`gamestate::GameState` trait**: the generic game interface (`to_tensor`, `legal_actions_mask`, `apply_action`, `get_terminal_value`, plus associated consts `N_CHAN/ROWS/COLS/ACTN_SPC_SIZE`). The engine and evaluators are generic over it; `pente::PenteState` is the only implementation. Actions are flat indices `row * COLS + col`.
- **Value perspective**: `get_terminal_value` is from the side-to-move's perspective. In Pente the player who just moved wins, so a terminal state returns `-1` (or `0` for a draw). `run_self_play` back-fills outcomes by walking states in reverse with alternating sign.
- **`evaluator::Evaluator` trait**: `evaluate(&G) -> (policy probs, value)`. Policy is a **probability** distribution (not log) with illegal moves masked out by `utils::legalize_policy`. Implementations: `NullEvaluator` (uniform), `TardEvaluator` (direct batch-of-1 forward pass; used by the `inference` bin and `SingleThreaded`), `BatchedClient`, and `ReplicatedClient`. To add a new threaded strategy, implement `EvalBackend` and add a `Mode` variant in `main.rs`.
- **MCTS (`engine.rs`)**: arena-allocated `MctsTree` (root is `NodeId` 0, children as first-child/next-sibling linked lists), PUCT selection, Dirichlet noise added to root priors (`DirichletHParams`; `epsilon=0` disables it). `mcts()` returns **log** posterior probabilities; callers `exp()` before sampling. The model's policy head also outputs log-softmax.
- **Model (`tard.rs`)**: `Tard`, a 128-channel conv trunk with 10 residual blocks plus policy and value heads. `compute_loss` = cross-entropy against the MCTS posterior + MSE against the game outcome.

## Gotchas

- `OutputManager::new("output", false)` in `train_tard` **deletes the entire `output/` directory** when training starts. Copy any checkpoints you want to keep out of `output/models/` first.
- Training resumes from, and `inference` loads, `./model.bpk` in the repo root. `utils::load_model` ignores its `path` argument and always reads `"model.bpk"`. To continue from a checkpoint, copy `output/models/model_<step>.bpk` to `model.bpk`.
- `*.bpk`, `output/`, `*.data`, and `*.svg` are gitignored. Model files and `perf.data` (several GB) live untracked in the repo root.
- `GameState::apply_action` with an illegal action is UB by contract; only a `debug_assert!` guards it.
