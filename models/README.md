# Saved models

Trained models worth keeping, copied out of `output/` (which is gitignored and
holds every checkpoint of every run). Unlike the rest of the repo, the `.bpk`
weights here are committed.

Each model lives in its game's directory as a set of files sharing one name:

| File | What it is |
|---|---|
| `<name>.bpk` | the weights |
| `<name>.net.json` | its sidecar: the game and network size. Required: a model can't be loaded without it |
| `<name>.train.json` | the training run's `config.json` (how it was trained) |
| `<name>.metrics.csv` | the training run's `metrics.csv` (loss and throughput per iteration, up to the end of the run) |

Names are `<game>-<training steps>` (e.g. `minipente-200k`), plus a suffix if
two models would collide. Avoid dots in names: the sidecar's path is the
model's with its extension swapped for `.net.json`.

Use them like any model file:

```bash
cargo run --release -- play --model models/minipente/minipente-200k.bpk
cargo run --release -- arena --model-a models/minipente/minipente-200k.bpk --model-b output/run_<ts>/checkpoints/step_<N>/model.bpk
cargo run --release -- web    # lists them under "models/<game>"
```

To save another one from a run's checkpoint:

```bash
ckpt=output/run_<ts>/checkpoints/step_<N>; name=models/<game>/<game>-<steps>
cp $ckpt/model.bpk $name.bpk
cp $ckpt/model.net.json $name.net.json
cp $ckpt/../../config.json $name.train.json
cp $ckpt/../../metrics.csv $name.metrics.csv
```

Then add a row below.

## Models

| Model | Game | Network | Trained | Source |
|---|---|---|---|---|
| `minipente/minipente-200k` | MiniPente | 64 channels, 6 blocks, 64 value hidden | 200,000 steps (~10.6 h), batched self-play with 512 workers and 400 rollouts; final loss 1.80 (policy 1.45, value 0.34), mean game 25.7 plies, no draws | `output/run_1794483901/checkpoints/step_200000` |
