# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Tard" is an AlphaZero-style AI for the board game Pente (9x9 board, capture rules), written in Rust using the `burn` deep learning framework (crate name `pentard`). It trains a policy/value ResNet via MCTS self-play.

## Commands

```bash
cargo build --release                      # build (use --release for anything performance-sensitive)
cargo run --release --bin pentard -- --mode batched   # self-play training (src/main.rs); --mode single|batched|replicated, also --workers, --rollouts, --temp-moves, --eval-batch-size, --model, --start-step, --output-dir
cargo run --release --bin inference -- --model model.bpk   # play against a trained model (default ./model.bpk)
cargo run --release --bin interactive      # human-vs-human Pente in the terminal (no AI)
```

- Leave `CUDA_PATH` unset: cubecl then searches `/usr/local/cuda`, `/opt/cuda`, `/usr/lib/cuda`, `/usr` for `nvvm/libdevice/libdevice.10.bc`. Pointing it at a path without that file (e.g. `/usr/local/cuda`, which doesn't exist here) makes GPU kernels fail at the first tensor readback with "libdevice.10.bc was not found". burn is built with the `cuda`, `fusion`, and `flex` features; `Device::default()` picks the backend.
- Logging uses `tracing` with `EnvFilter`; defaults are `warn` globally and `pentard=info`. Override with e.g. `RUST_LOG=pentard=debug`.
- Profiling: `cargo flamegraph --dev --min-width 0.0 -b pentard` (requires `sudo sysctl -w kernel.perf_event_paranoid=-1` and `kernel.kptr_restrict=0`).
- There is no test suite. `src/bin/burn_test.rs` and `src/bin/crossbeam_test.rs` are scratch experiments, not tests.

## Architecture

The training loop lives in `src/main.rs` (`train_tard`). Everything is configured by CLI args (`--help` lists them): self-play mode/workers/rollouts, and the training hyperparameters (`--total-steps`, `--steps-per-iter`, `--batch-size`, `--max-lr`/`--min-lr`, `--dirichlet-epsilon`/`--dirichlet-alpha`, replay-buffer sizing, etc.) with defaults matching the original hardcoded values. Network size is fixed by the `TRUNK_CHANNELS`/`RES_BLOCKS`/`VALUE_HIDDEN` consts in `tard.rs`, since checkpoints only load into a same-sized network. `loss.csv` records the mean loss over each iteration's steps. At startup, `utils::OutputManager::new(output_dir)` creates a fresh `<output-dir>/run_<unix_seconds>/` directory (it never deletes or reuses earlier runs), and the model is loaded from `--model` if that file exists, otherwise freshly initialized. Each iteration:
1. Pulls finished games from a `Box<dyn selfplay::SelfPlaySource<G>>` (`next_game`) into a `tard::TrainingBuffer` (bounded replay buffer).
2. Runs `n_training_steps_per_iter` AdamW steps with cosine-annealed LR.
3. Pushes new weights back with `source.update_model(&tard)`, saves `<run_dir>/models/model_<step>.bpk` plus the AdamW state as `optim_<step>.bpk`, and appends to `<run_dir>/loss.csv`.

Self-play sources (`--mode`), in `src/selfplay.rs`:
- `SingleThreaded`: runs `engine::run_self_play` inline with a local `TardEvaluator`.
- `WorkerPool<G, B: EvalBackend<G>>`: N worker threads, each with an `Evaluator` from `backend.client()`, sending games over a crossbeam channel. `Drop` closes the channel and joins the workers. Backends live in `src/evaluator/`:
  - `BatchedBackend` (`batched.rs`): workers send requests to one server thread, which batches up to `--eval-batch-size` positions per GPU forward pass. Clients send the already-encoded `to_tensor()` input and legalize the returned policy themselves; each request carries its own one-shot response channel so a dead server makes `recv` fail rather than hang. Weight updates go to the server thread over a channel.
  - `ReplicatedBackend` (`replicated.rs`): each worker owns a model copy. `update` writes into a shared `Mutex<Tard>` and bumps an `AtomicU64` version; a client re-clones the model on its next `evaluate` after the version changes.
- Models handed to evaluators are always `tard.valid()` (the non-autodiff model).

Key abstractions:
- **`gamestate::GameState` trait**: the generic game interface (`to_tensor`, `legal_actions_mask` -> `Vec<bool>`, `apply_action`, `get_terminal_value` -> `f32`, plus associated consts `N_CHAN/ROWS/COLS/ACTN_SPC_SIZE`). The engine and evaluators are generic over it; `pente::PenteState` is the only implementation. Actions are flat indices `row * COLS + col`.
- **Value perspective**: `get_terminal_value` is from the side-to-move's perspective. In Pente the player who just moved wins, so a terminal state returns `-1` (or `0` for a draw). `run_self_play` back-fills outcomes by walking states in reverse with alternating sign.
- **`evaluator::Evaluator` trait**: `evaluate(&G) -> (policy probs, value)`. Policy is a **probability** distribution (not log) with illegal moves masked out by `utils::legalize_policy`. Implementations: `NullEvaluator` (uniform), `TardEvaluator` (direct batch-of-1 forward pass; used by the `inference` bin and `SingleThreaded`), `BatchedClient`, and `ReplicatedClient`. To add a new threaded strategy, implement `EvalBackend` and add a `Mode` variant in `main.rs`.
- **MCTS (`engine.rs`)**: arena-allocated `MctsTree` (root is `NodeId` 0, children as first-child/next-sibling linked lists), PUCT selection, Dirichlet noise added to root priors (`DirichletHParams`; `epsilon=0` disables it). `mcts()` returns the posterior as plain probabilities (share of root visits), which is also the stored training target. The model's policy head outputs **log**-softmax; evaluators `exp()` it and `compute_loss` multiplies it directly by the posterior.
- **Model (`tard.rs`)**: `Tard`, a 128-channel conv trunk with 10 residual blocks plus policy and value heads; `Tard::for_game::<G>()` and `utils::load_model::<G>()` size it from a `GameState`. `compute_loss` = cross-entropy against the MCTS posterior + MSE against the game outcome.

## Notes

- To resume training from a checkpoint, pass it directly: `--model output/run_<ts>/models/model_<step>.bpk`. The step is parsed from the `model_<N>` filename (override with `--start-step`); a sibling `optim_<N>.bpk` restores the optimizer and the cosine LR schedule is fast-forwarded to step N. The replay buffer is not persisted, so a resumed run re-does the warm-up fill. Training stops at `n_training_steps`. If the `--model` file exists but fails to load (corrupt, or the architecture changed), training panics rather than silently starting fresh; if it doesn't exist, training starts from scratch.
- Older runs (before per-run directories) left checkpoints directly in `output/models/` and `output/loss.csv`; these are no longer written to.
- `*.bpk`, `output/`, `*.data`, and `*.svg` are gitignored. Model files and `perf.data` (several GB) live untracked in the repo root.
- `PenteState::apply_action` panics (`assert!`, in release builds too) on an illegal action. Callers should pick actions from `legal_actions_mask` or check `is_legal_action`; `PenteState::play(row, col)` is the non-panicking variant that returns `false` instead.
