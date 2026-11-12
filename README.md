AI for board game Pente

## Usage

```bash
cargo run --release --bin pentard -- --mode batched        # self-play training
cargo run --release --bin inference -- --model model.bpk   # play against a trained model
cargo run --release --bin arena -- --model-a a.bpk --model-b b.bpk   # match two models (--games, --rollouts[-a|-b], --temp-moves, --threads, --show, --watch [--delay-ms])
cargo run --release --bin interactive                      # human-vs-human Pente
```

Set `RUST_BACKTRACE=1` to get backtraces on panics.

## Profiling

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
sudo sysctl -w kernel.kptr_restrict=0
cargo flamegraph --dev --min-width 0.0 -b pentard
```
