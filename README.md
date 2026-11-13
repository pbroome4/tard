AlphaZero-style AI for board games. Currently implemented: Pente (official rules: 19x19, tournament opening rule), MiniPente (Pente on a 9x9 board) and Hive (base game).

## Usage

```bash
cargo run --release --bin tard -- --game pente --mode batched   # self-play training (--game minipente|pente|hive)
cargo run --release --bin inference -- --model model.bpk   # play against a trained model
cargo run --release --bin arena -- --model-a a.bpk --model-b b.bpk   # match two models (--games, --rollouts[-a|-b], --temp-moves, --threads, --show, --watch [--delay-ms])
cargo run --release --bin interactive -- --game pente     # human-vs-human (--game minipente|pente|hive)
cargo run --release --bin web -- --model model.bpk          # play MiniPente against a model in your browser (http://127.0.0.1:8080)
```

Set `RUST_BACKTRACE=1` to get backtraces on panics.

## Profiling

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
sudo sysctl -w kernel.kptr_restrict=0
cargo flamegraph --dev --min-width 0.0 -b tard
```
