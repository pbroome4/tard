# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Tard" is an AlphaZero-style AI for two-player board games, currently Pente (official: 19x19, tournament opening rule) and MiniPente (9x9 Pente), written in Rust using the `burn` deep learning framework (crate name `tard`). It trains a policy/value ResNet via MCTS self-play.

## Commands

```bash
cargo build --release                      # build (use --release for anything performance-sensitive)
cargo run --release --bin tard -- --mode batched   # self-play training (src/main.rs); --game minipente|pente, --mode single|batched|replicated, also --workers, --rollouts, --temp-moves, --eval-batch-size, --resume, --output-dir
cargo run --release --bin inference -- --model model.bpk   # play against a trained model (default ./model.bpk)
cargo run --release --bin arena -- --model-a a.bpk --model-b b.bpk   # model-vs-model match: W/L/D, per-color results, Elo estimate (--games 20, --rollouts 800 or --rollouts-a/-b, --temp-moves 4, --threads 4, --show; --watch prints every move live, one game at a time, --delay-ms paces it)
cargo run --release --bin interactive -- --game pente   # human-vs-human in the terminal (no AI); --game minipente (default) or pente
```

- burn is built with the `cuda`, `fusion`, and `flex` features; `Device::default()` picks the backend.
- Logging uses `tracing` with `EnvFilter`; defaults are `warn` globally and `tard=info`. Override with e.g. `RUST_LOG=tard=debug`.
- Profiling: `cargo flamegraph --dev --min-width 0.0 -b tard` (requires `sudo sysctl -w kernel.perf_event_paranoid=-1` and `kernel.kptr_restrict=0`).
- Tests: `cargo test` (about 3 s, CPU only). Unit tests live in `#[cfg(test)]` modules next to the code: game rules (`minipente.rs` and `pente.rs`, each with a `board(&["BWW."], black_to_move)` diagram helper; Pente also covers the tournament rule and seeded random playouts), MCTS/self-play invariants (`engine.rs`), utilities and the run-directory layout (`utils.rs`), worker-pool panic handling with fake backends (`selfplay.rs`), and the replay buffer (`tard.rs`). Nothing touching the network or GPU is unit-tested; for that, do a tiny real run, e.g. `--workers 16 --rollouts 10 --steps-per-iter 8 --batch-size 32 --warmup-iters 1 --total-steps 16`.
- `src/bin/burn_test.rs` and `src/bin/crossbeam_test.rs` are scratch experiments, not tests.

## Architecture

The training loop lives in `src/main.rs` (`train_tard`). Everything is configured by CLI args (`--help` lists them): self-play mode/workers/rollouts, and the training hyperparameters (`--total-steps`, `--steps-per-iter`, `--batch-size`, `--max-lr`/`--min-lr`, `--dirichlet-epsilon`/`--dirichlet-alpha`, replay-buffer sizing, etc.) with defaults matching the original hardcoded values. Network size is a `tard::NetConfig` (channels, residual blocks, value-head hidden units). Each game defines its own default for a fresh network (`minipente::default_net_config()`, `pente::default_net_config()`). MiniPente: 64/6/64, shrunk from the older, fairly arbitrary 128/10/128 to be ~6-7× cheaper per evaluation. Pente: 128/10/128, so the 21 stacked 3×3 convs give every point a view of the whole 19×19 board. Neither has been compared against other sizes, so treat them as starting points, not tuned values. `--channels`/`--blocks`/`--value-hidden` override it for a fresh network only; a loaded model keeps its saved size, and flags that disagree with it are an error. At startup the model is loaded from `--resume` (a checkpoint directory) or freshly initialized, and only then does `utils::OutputManager::new` create a fresh run directory (it never deletes or reuses earlier runs):

```
<output-dir>/run_<unix_seconds>/
  config.json                  # all CLI args (serde-serialized `Args`)
  metrics.csv                  # per iteration: step, wall_time_s, lr, loss, policy_loss, value_loss,
                               #   games, positions, mean_game_len, draw_rate (losses are means over the iteration),
                               #   then throughput: iter_time_s, selfplay_wait_s, train_s, positions_per_s,
                               #   evals_per_s, mean_eval_batch, eval_busy_frac
  checkpoints/step_<NNNNNN>/   # zero-padded step
    model.bpk                  # weights
    model.net.json             # its game + NetConfig (sidecar written by utils::save_model)
    optim.bpk                  # AdamW state
```

Each iteration:
1. Pulls finished games from a `Box<dyn selfplay::SelfPlaySource<G>>` (`next_game`) into a `tard::TrainingBuffer` (bounded replay buffer).
2. Runs `n_training_steps_per_iter` AdamW steps with cosine-annealed LR.
3. Pushes new weights back with `source.update_model(&tard)`, saves `checkpoints/step_<step>/`, and appends a row to `metrics.csv`.

Self-play sources (`--mode`), in `src/selfplay.rs`:
- `SingleThreaded`: runs `engine::run_self_play` inline with a local `TardEvaluator`.
- `WorkerPool<G, B: EvalBackend<G>>`: N worker threads, each with an `Evaluator` from `backend.client()`, sending games over a crossbeam channel. `Drop` closes the channel and joins the workers. Backends live in `src/evaluator/`:
  - `BatchedBackend` (`batched.rs`): workers send requests to one server thread, which batches up to `--eval-batch-size` positions per GPU forward pass. Clients send the already-encoded `to_tensor()` input and legalize the returned policy themselves; each request carries its own one-shot response channel so a dead server makes `recv` fail rather than hang. Weight updates go to the server thread over a channel.
  - `ReplicatedBackend` (`replicated.rs`): each worker owns a model copy. `update` writes into a shared `Mutex<Tard>` and bumps an `AtomicU64` version; a client re-clones the model on its next `evaluate` after the version changes.
- Models handed to evaluators are always `tard.valid()` (the non-autodiff model).

Key abstractions:
- **`gamestate::GameState` trait**: the generic game interface (`to_tensor`, `legal_actions_mask` -> `Vec<bool>`, `apply_action`, `get_terminal_value` -> `f32`, plus associated consts `NAME/N_CHAN/ROWS/COLS/ACTN_SPC_SIZE`). The engine, evaluators, self-play, model and trainer are generic over it; there are two implementations, each a separate, self-contained file with its own `Piece`, board and rules (they deliberately share no code): `minipente::MiniPenteState` (`--game minipente`: 9x9, Black first, no opening rule) and `pente::PenteState` (`--game pente`: official Pente, 19x19, White first, tournament rule: White's first stone on the center, White's second at least 3 intersections away, outside the central 5x5). Both number actions `row * COLS + col`; other games may number actions however they like. MiniPente's behavior and network input must stay unchanged so existing MiniPente models keep working.
- **Adding a game**: implement `GameState` (unique `NAME`; `Default` is the start position), then in `main.rs` add a `Game` variant and a `main()` match arm `train_tard::<NewGame>(&args, <its default NetConfig>)`. To use it from the terminal bins, also implement `gamestate::GridGame` (player names, `play_at(row, col)`, `action_coord`) and add it to their game dispatch. The generic code assumes: two players who strictly alternate turns (MCTS backup and self-play outcomes flip sign every ply); every non-terminal state has at least one legal action (make "pass" an explicit action if needed); deterministic perfect information; input shaped `N_CHAN x ROWS x COLS` for the conv trunk; `get_terminal_value` from the side-to-move's perspective. The `inference`, `arena` and `interactive` bins are generic over `G: GridGame` (row,col move entry): `inference` and `arena` pick the game from the model's `.net.json` game name, `interactive` takes `--game`.
- **Value perspective**: `get_terminal_value` is from the side-to-move's perspective. In both Pente games the player who just moved wins, so a terminal state returns `-1` (or `0` for a draw). `run_self_play` back-fills outcomes by walking states in reverse with alternating sign.
- **Throughput counters**: `evaluator::EvalStats` (shared atomics) counts positions evaluated, forward passes, and time inside forward passes; `TardEvaluator` and the batched server record into it, and `SelfPlaySource::eval_stats()` exposes it so `main.rs` can diff it per iteration. In batched mode the batch size is capped by `--workers`, since each worker has one request in flight.
- **`evaluator::Evaluator` trait**: `evaluate(&G) -> (policy probs, value)`. Policy is a **probability** distribution (not log) with illegal moves masked out by `utils::legalize_policy`. Implementations: `NullEvaluator` (uniform), `TardEvaluator` (direct batch-of-1 forward pass; used by the `inference` and `arena` bins and `SingleThreaded`), `BatchedClient`, and `ReplicatedClient`. To add a new threaded strategy, implement `EvalBackend` and add a `Mode` variant in `main.rs`.
- **MCTS (`engine.rs`)**: arena-allocated `MctsTree` (root is `NodeId` 0, children as first-child/next-sibling linked lists), PUCT selection, Dirichlet noise mixed into the root's child priors at the start of each search (`DirichletHParams`; `epsilon=0` disables it). Self-play reuses the chosen move's subtree (`MctsTree::reroot`), and `search` only runs enough simulations to bring the root up to `--rollouts` visits; `mcts()` always starts from a fresh tree. `mcts()` returns the posterior as plain probabilities (share of root visits), which is also the stored training target. The model's policy head outputs **log**-softmax; evaluators `exp()` it and `compute_loss` multiplies it directly by the posterior.
- **Model (`tard.rs`)**: `Tard`, a conv trunk of residual blocks plus policy and value heads. The trunk/value-head size comes from a `NetConfig`; input channels and the policy output come from the `GameState` (`Tard::for_game::<G>(&net, device)`). `utils::save_model::<G>` writes the weights plus a `<name>.net.json` sidecar (`ModelMeta`: game name + NetConfig, flat JSON), and `utils::load_model::<G>()` reads it, refuses a model trained for a different game, and rebuilds the right shape before loading. The sidecar is required; a model file without one can't be loaded. `compute_loss` = cross-entropy against the MCTS posterior + MSE against the game outcome. The heads avoid plain ReLU after their narrow 1-2 channel convs (policy uses `leaky_relu`; value is linear into `value_fc1`, then `leaky_relu`): a value head with a ReLU there died within the first 512 steps, outputting one constant for every position (value loss stuck at ~1.0).

## Notes

- To continue a run: `--resume output/run_<ts>/checkpoints/step_<N>`. It restores the weights and optimizer, takes N from the directory name, and fast-forwards the cosine LR schedule to step N. The replay buffer is not persisted, so a resumed run re-does the warm-up fill. Training stops at `--total-steps`.
- Without `--resume`, training starts from a fresh network. If a `--resume` checkpoint fails to load (corrupt, or for another game), training panics rather than silently starting fresh. The `inference` and `arena` bins take model files directly (each needs its `.net.json` sidecar next to it), e.g. `--model output/run_<ts>/checkpoints/step_<N>/model.bpk`.
- `*.bpk`, `output/`, `*.data`, and `*.svg` are gitignored. Model files and `perf.data` (several GB) live untracked in the repo root.
- `MiniPenteState::apply_action` and `PenteState::apply_action` panic (`assert!`, in release builds too) on an illegal action. Callers should pick actions from `legal_actions_mask` or check `is_legal_action`; their `play(row, col)` is the non-panicking variant that returns `false` instead.
- Pente is much heavier than MiniPente: each replay-buffer position holds 4×361 + 361 floats (~7 KB), so the default buffer (`--buffer-iters 10` × 32,768 positions) needs ~2.4 GB of RAM, and every evaluation runs the bigger network on a 19×19 board.
