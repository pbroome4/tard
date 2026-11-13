AlphaZero-style AI for board games. Currently implemented: Pente (official rules: 19x19, tournament opening rule), MiniPente (Pente on a 9x9 board), Hive (base game) and Ultimate Tic-Tac-Toe.

## Usage

Everything is one command, `tard`, with subcommands (add `--help` to any of them for its options):

```bash
cargo run --release -- train --game pente           # train by self-play (--game minipente|pente|hive|uttt)
cargo run --release -- play --model model.bpk       # play a trained model in the terminal
cargo run --release -- play --game hive             # human vs human in the terminal (--game minipente|pente|hive|uttt)
cargo run --release -- web                          # browser game room (http://127.0.0.1:8080): pick a game, then play a model, a friend, or watch two models
cargo run --release -- arena --model-a a.bpk --model-b b.bpk   # match two models (--games, --rollouts[-a|-b], --temp-moves, --threads, --show, --watch [--delay-ms])
```

Set `RUST_BACKTRACE=1` to get backtraces on panics.

## Code layout

- `src/main.rs`, `src/commands/`: the `tard` command, one module per subcommand
- `src/games/`: the game interface and one file per game
- `src/mcts.rs`, `src/selfplay.rs`, `src/evaluator/`: search and self-play
- `src/model.rs`, `src/training/`: the network, model files, replay buffer, loss and run output
- `web/index.html`: the page `tard web` serves

## Profiling

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
sudo sysctl -w kernel.kptr_restrict=0
cargo flamegraph --dev --min-width 0.0 -- train
```
