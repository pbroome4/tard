# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Tard" is an AlphaZero-style AI for two-player board games, currently Pente (official: 19x19, tournament opening rule), MiniPente (9x9 Pente), Hive (base game) and Ultimate Tic-Tac-Toe, written in Rust using the `burn` deep learning framework (crate name `tard`). It trains a policy/value ResNet via MCTS self-play.

## Commands

Everything is one binary, `tard`, with subcommands (`cargo run --release -- <command> --help` lists each one's flags):

```bash
cargo build --release                               # build (use --release for anything performance-sensitive)
cargo run --release -- train --game minipente       # self-play training; --game minipente|pente|hive|uttt, --mode single|batched|replicated, also --workers, --rollouts, --temp-moves, --eval-batch-size, --resume, --output-dir and the hyperparameters
cargo run --release -- play --model model.bpk       # play a model in the terminal (game from its sidecar); without --model, human vs human: --game minipente (default)|pente|hive|uttt
cargo run --release -- web                          # browser game room at http://127.0.0.1:8080: any game vs a model, vs a person on the same screen, or model vs model (--models <files/dirs> to search, default .; --rollouts 3000, --port, --host)
cargo run --release -- arena --model-a a.bpk --model-b b.bpk   # model-vs-model match: W/L/D, per-color results, Elo estimate (--games 20, --rollouts 800 or --rollouts-a/-b, --temp-moves 4, --threads 4, --show; --watch prints every move live, one game at a time, --delay-ms paces it)
```

- burn is built with the `cuda`, `fusion`, and `flex` features; `Device::default()` picks the backend.
- Logging uses `tracing` with `EnvFilter`, set up once in `src/main.rs`; defaults are `warn` globally and `tard=info`. Override with e.g. `RUST_LOG=tard=debug`.
- Profiling: `cargo flamegraph --dev --min-width 0.0 -- train ...` (requires `sudo sysctl -w kernel.perf_event_paranoid=-1` and `kernel.kptr_restrict=0`).
- Tests: `cargo test` (about 3 s, CPU only). Unit tests live in `#[cfg(test)]` modules next to the code: game rules (`games/minipente.rs` and `games/pente.rs`, each with a `board(&["BWW."], black_to_move)` diagram helper; Pente also covers the tournament rule and seeded random playouts; `games/hive.rs` builds positions from `(player, piece, q, r)` lists and checks each piece's movement, the placement/queen/one-hive/freedom-of-movement rules, win/draw/pass, and seeded random playouts that round-trip every move through its text notation; `games/uttt.rs` sets positions directly and checks forced boards, freed moves, won/tied small boards, the big-board win and draw, and seeded random playouts), the game list (`games/mod.rs`), MCTS invariants (`mcts.rs`), self-play games and worker-pool panic handling with fake backends (`selfplay.rs`), model sidecars (`model.rs`), policy legalizing (`evaluator/mod.rs`), the replay buffer and run-directory layout (`training/`), and the web server's replay, per-game views and model scanning (`commands/web.rs`). Nothing touching the network or GPU is unit-tested; for that, do a tiny real run, e.g. `train --workers 16 --rollouts 10 --steps-per-iter 8 --batch-size 32 --warmup-iters 1 --total-steps 16`.

## Style

- Formatting is rustfmt's default (`cargo fmt`; check with `cargo fmt --check`). Run it before committing.
- Functions return with an explicit `return x;`, never a bare tail expression (closures may stay expressions, e.g. `|x| x + 1`).
- Comments are plain `//`, including the ones describing a function, struct or constant. `///` is used only where clap reads it as `--help` text (fields of the `Args` structs, `Command` and `Game` variants), and `//!` for module headers.
- The game files follow one shape: types, the state struct (fields end in `_`), `default_net_config()`, `Default`, `impl GameState`, the inherent impl (`new`, `is_legal_action`, private helpers like `active_piece`), free functions (`coord_2_action`/`action_2_coord` for grid games), then `TextGame`, `WebGame`, `Display`, and tests. Actions are typed `Action`; a game's status enum is `Ongoing`/`Won`/`Draw`.

## Layout

```
src/main.rs           the `tard` CLI: parses the subcommand, sets up logging
src/commands/         one module per subcommand (Args + run): train, play, web, arena;
                      mod.rs has game_of_model (a model's game, from its sidecar)
src/lib.rs            the library the commands are built on:
  games/              mod.rs: GameState, TextGame, WebGame, the Game list and with_game!;
                      minipente.rs, pente.rs, hive.rs, uttt.rs: one self-contained file per game
  mcts.rs             the search (MctsTree, PUCT, root noise), best_move, select_action
  selfplay.rs         run_self_play (one game) and SelfPlaySource (SingleThreaded, WorkerPool)
  evaluator/          Evaluator/EvalBackend traits, legalize_policy, EvalStats; batched.rs, replicated.rs
  model.rs            the network (Tard, NetConfig) and model files (save_model/load_model, .net.json sidecar)
  training/           replay.rs (TrainingElement, TrainingBuffer), loss.rs (compute_loss), run_dir.rs (OutputManager, metrics)
  util.rs             normalize, panic_message
web/index.html        the `web` page, compiled in with include_str! (rebuild after editing it)
models/               saved trained models, committed (weights included); see models/README.md
```

## Architecture

The training loop is `train_tard` in `src/commands/train.rs`. Everything is configured by CLI args (`train --help` lists them): self-play mode/workers/rollouts, and the training hyperparameters (`--total-steps`, `--steps-per-iter`, `--batch-size`, `--max-lr`/`--min-lr`, `--dirichlet-epsilon`/`--dirichlet-alpha`, replay-buffer sizing, etc.) with defaults matching the original hardcoded values. Network size is a `model::NetConfig` (channels, residual blocks, value-head hidden units). Each game defines its own default for a fresh network (`Game::default_net()`, which calls the game module's `default_net_config()`). MiniPente: 64/6/64, shrunk from the older, fairly arbitrary 128/10/128 to be ~6-7× cheaper per evaluation. Pente: 128/10/128, so the 21 stacked 3×3 convs give every point a view of the whole 19×19 board. Hive: 64/8/64. Ultimate Tic-Tac-Toe: 64/6/64, like the other 9x9 game. None has been compared against other sizes, so treat them as starting points, not tuned values. `--channels`/`--blocks`/`--value-hidden` override it for a fresh network only; a loaded model keeps its saved size, and flags that disagree with it are an error. At startup the model is loaded from `--resume` (a checkpoint directory) or freshly initialized, and only then does `training::OutputManager::new` create a fresh run directory (it never deletes or reuses earlier runs):

```
<output-dir>/run_<unix_seconds>/
  config.json                  # all `train` args (serde-serialized `commands::train::Args`)
  metrics.csv                  # per iteration: step, wall_time_s, lr, loss, policy_loss, value_loss,
                               #   games, positions, mean_game_len, draw_rate (losses are means over the iteration),
                               #   then throughput: iter_time_s, selfplay_wait_s, train_s, positions_per_s,
                               #   evals_per_s, mean_eval_batch, eval_busy_frac
  checkpoints/step_<NNNNNN>/   # zero-padded step
    model.bpk                  # weights
    model.net.json             # its game + NetConfig (sidecar written by model::save_model)
    optim.bpk                  # AdamW state
```

Each iteration:
1. Pulls finished games from a `Box<dyn selfplay::SelfPlaySource<G>>` (`next_game`) into a `training::TrainingBuffer` (bounded replay buffer).
2. Runs `n_training_steps_per_iter` AdamW steps with cosine-annealed LR.
3. Pushes new weights back with `source.update_model(&tard)`, saves `checkpoints/step_<step>/`, and appends a row to `metrics.csv`.

Self-play sources (`--mode`), in `src/selfplay.rs`:
- `SingleThreaded`: runs `selfplay::run_self_play` inline with a local `TardEvaluator`.
- `WorkerPool<G, B: EvalBackend<G>>`: N worker threads, each with an `Evaluator` from `backend.client()`, sending games over a crossbeam channel. `Drop` closes the channel and joins the workers. Backends live in `src/evaluator/`:
  - `BatchedBackend` (`batched.rs`): workers send requests to one server thread, which batches up to `--eval-batch-size` positions per GPU forward pass. Clients send the already-encoded `to_tensor()` input and legalize the returned policy themselves; each request carries its own one-shot response channel so a dead server makes `recv` fail rather than hang. Weight updates go to the server thread over a channel.
  - `ReplicatedBackend` (`replicated.rs`): each worker owns a model copy. `update` writes into a shared `Mutex<Tard>` and bumps an `AtomicU64` version; a client re-clones the model on its next `evaluate` after the version changes.
- Models handed to evaluators are always `tard.valid()` (the non-autodiff model).

Key abstractions:
- **`games::GameState` trait**: the generic game interface (`to_tensor`, `legal_actions_mask` -> `Vec<bool>`, `apply_action`, `get_terminal_value` -> `f32`, plus associated consts `NAME/N_CHAN/ROWS/COLS/ACTN_SPC_SIZE`). MCTS, evaluators, self-play, the model and the trainer are generic over it; each game is a separate, self-contained file in `src/games/` (they deliberately share no code): `minipente::MiniPenteState` (`--game minipente`: 9x9, Black first, no opening rule), `pente::PenteState` (`--game pente`: official Pente, 19x19, White first, tournament rule: White's first stone on the center, White's second at least 3 intersections away, outside the central 5x5), `hive::HiveState` (`--game hive`), and `uttt::UtttState` (`--game uttt`: ultimate tic-tac-toe, X first; your cell picks the small board the opponent must play in, or any open board if that one is decided; no moves in decided boards; three won small boards in a row wins, all boards decided without that is a draw). Both Pente games and UTTT number actions `row * COLS + col` on a 9x9 (or 19x19) grid; UTTT's 5 input planes are my marks, their marks, and small boards won by me / won by them / tied; nothing encodes where the side to move may play (deliberately: the network is meant to learn the rules). Hive (base game: 11 pieces each, White first, tournament rule of no queen on the first turn, pass when stuck, draw at `hive::MAX_PLIES` = 200 plies) has no board, so the hive is re-centered after every move in a 24×24 frame of axial hex coordinates (cell `r * 24 + q`; 22 pieces plus a ring of neighbors always fit). Its action is `piece * 576 + cell` (piece 0-10 of the side to move: Q, S1-2, B1-2, G1-3, A1-3; placing and moving are the same action, and same-kind pieces are placed lowest number first) plus a final pass action, 6337 in all; its 37 input planes give each piece's position (so the policy can tell which piece an action moves), stack tops and heights, both hands, the turn number and the ply. MiniPente's behavior and network input must stay unchanged so existing MiniPente models keep working.
- **`games::TextGame` trait**: what the commands need beyond `GameState`: player names, `MOVE_FORMAT`, `parse_action` (only ever returns legal actions) and `action_text`. Moves are typed and printed as "row, col" for the Pente games and UTTT, and in standard Hive notation (`wA1 -bQ`, `wB1 bQ` for on top, `pass`) for Hive.
- **`games::WebGame` trait**: what `web` needs beyond `TextGame`: `position_json` (the whole position, tagged with a `kind` the page knows how to draw: `"grid"` for both Pente games with rows/cols/board/captures, `"hive"` with cells (axial q, r and the stack, bottom first) and hands, `"uttt"` with the 9x9 marks, each small board's state (`.`/`X`/`O`/`T` for tied) and the forced `next` board) and `action_json` (where an action plays: `{row, col}`, or Hive's `{piece, q, r}` / `{pass}`).
- **`games::Game` and `with_game!`**: the one list of games. `Game` is the `--game` value enum (also serialized into config.json), with `name()`/`from_name()` (the `GameState::NAME`) and `default_net()`. `with_game!(game, G => expr)` runs generic code with `G` as that game's state type; every command dispatches through it (`play` and `arena` find the game from the model sidecar with `commands::game_of_model`).
- **Adding a game**: write `src/games/<name>.rs` implementing `GameState` (unique `NAME`; `Default` is the start position) and `TextGame`, with a `default_net_config()`, then in `games/mod.rs` add the module, a `Game` variant (and to `Game::ALL`), and arms in `with_game!` and `Game::default_net`. Every command then supports it, except that `web/index.html` needs a renderer for a new `position_json` kind (and a title/description in `commands/web.rs`'s `about`). The generic code assumes: two players who strictly alternate turns (MCTS backup and self-play outcomes flip sign every ply); every non-terminal state has at least one legal action (make "pass" an explicit action if needed); deterministic perfect information; input shaped `N_CHAN x ROWS x COLS` for the conv trunk; `get_terminal_value` from the side-to-move's perspective.
- **Value perspective**: `get_terminal_value` is from the side-to-move's perspective. In both Pente games and UTTT the player who just moved wins, so a terminal state returns `-1` (or `0` for a draw). In Hive a player can lose on their own move (by surrounding their own queen), so `+1` also occurs. `run_self_play` back-fills outcomes by walking states in reverse with alternating sign.
- **Throughput counters**: `evaluator::EvalStats` (shared atomics) counts positions evaluated, forward passes, and time inside forward passes; `TardEvaluator` and the batched server record into it, and `SelfPlaySource::eval_stats()` exposes it so `train` can diff it per iteration. In batched mode the batch size is capped by `--workers`, since each worker has one request in flight.
- **`evaluator::Evaluator` trait**: `evaluate(&G) -> (policy probs, value)`. Policy is a **probability** distribution (not log) with illegal moves masked out by `evaluator::legalize_policy`. Implementations: `NullEvaluator` (uniform), `TardEvaluator` (direct batch-of-1 forward pass; used by `play`, `web`, `arena` and `SingleThreaded`), `BatchedClient`, and `ReplicatedClient`. To add a new threaded strategy, implement `EvalBackend` and add a `Mode` variant in `commands/train.rs`.
- **MCTS (`mcts.rs`)**: arena-allocated `MctsTree` (root is `NodeId` 0, children as first-child/next-sibling linked lists), PUCT selection, Dirichlet noise mixed into the root's child priors at the start of each search (`DirichletHParams`; `epsilon=0` disables it). Self-play (`selfplay::run_self_play`) reuses the chosen move's subtree (`MctsTree::reroot`), and `search` only runs enough simulations to bring the root up to `--rollouts` visits; `mcts()` and `best_move()` always start from a fresh tree. `mcts()` returns the posterior as plain probabilities (share of root visits), which is also the stored training target. The model's policy head outputs **log**-softmax; evaluators `exp()` it and `compute_loss` multiplies it directly by the posterior.
- **Model (`model.rs`)**: `Tard`, a conv trunk of residual blocks plus policy and value heads. The trunk/value-head size comes from a `NetConfig`; input channels and the policy output come from the `GameState` (`Tard::for_game::<G>(&net, device)`). `model::save_model::<G>` writes the weights plus a `<name>.net.json` sidecar (`ModelMeta`: game name + NetConfig, flat JSON), and `model::load_model::<G>()` reads it, refuses a model trained for a different game, and rebuilds the right shape before loading. The sidecar is required; a model file without one can't be loaded. `training::compute_loss` = cross-entropy against the MCTS posterior + MSE against the game outcome. The heads avoid plain ReLU after their narrow 1-2 channel convs (policy uses `leaky_relu`; value is linear into `value_fc1`, then `leaky_relu`): a value head with a ReLU there died within the first 512 steps, outputting one constant for every position (value loss stuck at ~1.0).

## Notes

- To continue a run: `train --resume output/run_<ts>/checkpoints/step_<N>`. It restores the weights and optimizer, takes N from the directory name, and fast-forwards the cosine LR schedule to step N. The replay buffer is not persisted, so a resumed run re-does the warm-up fill. Training stops at `--total-steps`.
- Without `--resume`, training starts from a fresh network. If a `--resume` checkpoint fails to load (corrupt, or for another game), training panics rather than silently starting fresh. `play`, `web` and `arena` take model files directly (each needs its `.net.json` sidecar next to it), e.g. `--model output/run_<ts>/checkpoints/step_<N>/model.bpk`.
- `*.bpk`, `output/`, `*.data`, and `*.svg` are gitignored, except `models/**/*.bpk`: saved models are committed. Other model files and `perf.data` (several GB) live untracked in the repo root.
- Saved models (`models/<game>/<name>.bpk`) keep their sidecar as `<name>.net.json` plus the run's `config.json` and `metrics.csv` as `<name>.train.json` / `<name>.metrics.csv`; `models/README.md` lists them and how to add one. `web` lists them first, by name, ahead of run checkpoints.
- Every game's `apply_action` panics (`assert!`, in release builds too) on an illegal action. Callers should pick actions from `legal_actions_mask` or check `is_legal_action` (or `TextGame::parse_action`, which only returns legal actions); the Pente games' `play(row, col)` is a non-panicking variant that returns `false` instead.
- Pente is much heavier than MiniPente: each replay-buffer position holds 4×361 + 361 floats (~7 KB), so the default buffer (`--buffer-iters 10` × 32,768 positions) needs ~2.4 GB of RAM, and every evaluation runs the bigger network on a 19×19 board.
- Hive is heavy on memory: each replay-buffer position holds 37×576 input floats plus a 6337-float posterior (~110 KB), so the default buffer (`--buffer-iters 10` × 32,768 positions) would need ~36 GB of RAM. Shrink it (`--buffer-iters`, or `--steps-per-iter`/`--batch-size`/`--sample-reuse`, which set the per-iteration size). Also pass `--dirichlet-alpha` (e.g. 0.3): the default, 20 / action-space size, is ~0.003 for Hive's 6337 actions, far below what its ~20-100 legal moves call for.
- `web` (`src/commands/web.rs`, a `tiny_http` server) is stateless: the page POSTs the game name and whole move list to `/api/play` (`ai: true` plus a `model` id asks for the computer's reply), and the server replays it with that game's rules (rejecting illegal moves) and runs `mcts` with the model. `GET /api/games` lists the games and their models: every `*.bpk` with a `.net.json` sidecar under `--models` (default `.`, skipping `target/` and hidden directories), rescanned on each call, grouped by run, newest step first. Requests may only name scanned models; up to 4 stay loaded. Computer-vs-computer games are driven by the page (one `/api/play` per move, with each side's model). It handles one request at a time and listens on 127.0.0.1 by default.
