//! The network (`Tard`: a conv trunk of residual blocks with policy and
//! value heads), its size (`NetConfig`), and saving and loading model files
//! with their `.net.json` sidecar.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::store::ModuleRecord;
use burn::tensor::activation::{leaky_relu, log_softmax, relu, tanh};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

use crate::games::GameState;

// Size of the network's trunk and value head. The input and output layers
// are sized from the GameState instead. A checkpoint only loads into a
// network with the same NetConfig, so it is saved next to every model file
// (see utils::save_model / utils::load_model).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetConfig {
    pub channels: usize,     // channels in the conv trunk
    pub blocks: usize,       // residual blocks in the trunk
    pub value_hidden: usize, // hidden units in the value head
}

// Negative slope of the leaky_relu activations in the policy and value heads.
const HEAD_LEAKY_SLOPE: f64 = 0.01;

#[derive(Debug, Module)]
pub struct Tard {
    // 1. Shared Feature Extractor (Convolutional Trunk)
    conv1: Conv2d, // initial convolutional block
    bn1: BatchNorm,
    resnets: Vec<ResNet>, // a couple of feed forward "residual networks"

    // 2. Policy Head (Decides which move to make)
    // Output log probability of move to make
    policy_conv: Conv2d,
    policy_bn: BatchNorm,
    policy_fc: Linear,

    // 3. Value Head (Evaluates who is winning: -1 to +1)
    value_conv: Conv2d,
    value_bn: BatchNorm,
    value_fc1: Linear,
    value_fc2: Linear,
}
impl Tard {
    // Builds a network for game G with the given trunk/value-head size.
    pub fn for_game<G: GameState>(net: &NetConfig, device: &Device) -> Self {
        return Self::new(G::N_CHAN, G::ROWS, G::COLS, G::ACTN_SPC_SIZE, net, device);
    }

    pub fn new(
        in_chan: usize,
        rows: usize,
        cols: usize,
        action_space_size: usize,
        net: &NetConfig,
        device: &Device,
    ) -> Self {
        info!(
            "Creating Tard in_chan:{}, rows: {},  cols:{}, {:?}",
            in_chan, rows, cols, net
        );
        let ch = net.channels;
        let conv1 = Conv2dConfig::new([in_chan, ch], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(ch).init(device);
        let mut resnets = Vec::new();
        for _ in 0..net.blocks {
            resnets.push(ResNet::new(ch, device));
        }

        let policy_conv = Conv2dConfig::new([ch, 2], [1, 1]).init(device);
        let policy_bn = BatchNormConfig::new(2).init(device);
        let policy_fc = LinearConfig::new(2 * rows * cols, action_space_size).init(device);

        let value_conv = Conv2dConfig::new([ch, 1], [1, 1]).init(device);
        let value_bn = BatchNormConfig::new(1).init(device);
        let value_fc1 = LinearConfig::new(rows * cols, net.value_hidden).init(device);
        let value_fc2 = LinearConfig::new(net.value_hidden, 1).init(device);

        let s = Self {
            conv1,
            bn1,
            resnets,
            policy_conv,
            policy_bn,
            policy_fc,
            value_conv,
            value_bn,
            value_fc1,
            value_fc2,
        };
        return s;
    }

    pub fn forward(&self, x: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        // 1. Convolutional Trunk
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let mut x = relu(x);
        for resnet in self.resnets.iter() {
            x = resnet.forward(x);
        }

        // 2. Policy Head
        // leaky_relu rather than relu so the narrow 2-channel layer can't
        // get stuck with every square zeroed (see the value head below).
        let p = self.policy_conv.forward(x.clone());
        let p = self.policy_bn.forward(p);
        let p = leaky_relu(p, HEAD_LEAKY_SLOPE);
        let [batch_size, channels, height, width] = p.shape().dims();
        let p = p.reshape([batch_size, channels * height * width]);
        let p = self.policy_fc.forward(p);
        let p = log_softmax(p, 1);

        // 3. Value Head
        // No hard ReLU anywhere in this head: with a single channel, one
        // shared ReLU threshold after value_bn could zero every square for
        // every position, cutting all gradient and leaving the head stuck
        // at a constant output. value_conv -> value_bn -> value_fc1 stays
        // linear and the hidden layer uses leaky_relu, so a gradient always
        // flows back. (The trunk avoids this through its residual skips.)
        let v = self.value_conv.forward(x);
        let v = self.value_bn.forward(v);
        let [batch_size, channels, height, width] = v.shape().dims();
        let v = v.reshape([batch_size, channels * height * width]);
        let v = self.value_fc1.forward(v);
        let v = leaky_relu(v, HEAD_LEAKY_SLOPE);
        let v = self.value_fc2.forward(v);
        let v = tanh(v);

        return (p, v);
    }
}

#[derive(Debug, Module)]
struct ResNet {
    conv1: Conv2d,
    conv2: Conv2d,
    bn1: BatchNorm,
    bn2: BatchNorm,
}

impl ResNet {
    fn new(num_chan: usize, device: &Device) -> Self {
        let conv1 = Conv2dConfig::new([num_chan, num_chan], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([num_chan, num_chan], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        let bn1 = BatchNormConfig::new(num_chan).init(device);
        let bn2 = BatchNormConfig::new(num_chan).init(device);
        let s = Self {
            conv1,
            conv2,
            bn1,
            bn2,
        };
        return s;
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let identity = x.clone();
        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = relu(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = relu(out + identity);
        return out;
    }
}

////// Model files //////

// What a model file's weights are shaped for, saved next to it as
// `<name>.net.json`: the game (input/output layers) and the NetConfig
// (trunk and value head). Serialized flat, e.g.
// `{"game": "minipente", "channels": 128, "blocks": 10, "value_hidden": 128}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMeta {
    pub game: String, // GameState::NAME
    #[serde(flatten)]
    pub net: NetConfig,
}

// Path of the sidecar for a model file: `.../model.bpk` -> `.../model.net.json`.
pub fn model_meta_path(model: &Path) -> PathBuf {
    return model.with_extension("net.json");
}

// Writes `meta` as the sidecar for `model`.
pub fn write_model_meta(model: &Path, meta: &ModelMeta) -> Result<(), Box<dyn Error>> {
    fs::write(model_meta_path(model), serde_json::to_string_pretty(meta)?)?;
    return Ok(());
}

// Reads the sidecar saved next to `model`. It is required: without it
// there's no way to know the network's shape.
pub fn read_model_meta(model: &Path) -> Result<ModelMeta, Box<dyn Error>> {
    let p = model_meta_path(model);
    let json = fs::read_to_string(&p)
        .map_err(|e| format!("can't read model sidecar {}: {}", p.display(), e))?;
    return Ok(serde_json::from_str(&json)?);
}

// Errors if `meta` records a game other than G.
pub fn check_game<G: GameState>(meta: &ModelMeta) -> Result<(), Box<dyn Error>> {
    if meta.game != G::NAME {
        return Err(format!(
            "model was trained for game '{}', not '{}'",
            meta.game,
            G::NAME
        )
        .into());
    }
    return Ok(());
}

// Saves the weights to `path` and their game and network size next to them.
pub fn save_model<G: GameState>(
    tard: &Tard,
    net: &NetConfig,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    tard.clone().save_file(path)?;
    write_model_meta(
        path,
        &ModelMeta {
            game: G::NAME.to_string(),
            net: *net,
        },
    )?;
    return Ok(());
}

// Loads a model saved by save_model into a network for game G.
// Returns the network and its size.
pub fn load_model<G: GameState>(
    p: &Path,
    device: &Device,
) -> Result<(Tard, NetConfig), Box<dyn Error>> {
    let meta = read_model_meta(p)?;
    check_game::<G>(&meta)?;
    info!("Loading model record: {} ({:?})", p.display(), meta);
    let record = ModuleRecord::load(p)?;
    let tard = Tard::for_game::<G>(&meta.net, device).load_record(record);
    info!("Tard model loaded");
    return Ok((tard, meta.net));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::minipente::MiniPenteState;

    // A fresh, empty scratch directory under the system temp dir.
    fn scratch_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tard_test_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        return d;
    }

    #[test]
    fn model_meta_sidecar_sits_next_to_the_model_file() {
        assert_eq!(
            model_meta_path(Path::new("ckpt/step_000512/model.bpk")),
            PathBuf::from("ckpt/step_000512/model.net.json")
        );
        assert_eq!(
            model_meta_path(Path::new("model")),
            PathBuf::from("model.net.json")
        );
    }

    #[test]
    fn model_meta_roundtrips_and_is_required() {
        let d = scratch_dir("netcfg");
        let model = d.join("model.bpk");
        assert!(
            read_model_meta(&model).is_err(),
            "a missing sidecar is an error"
        );
        let meta = ModelMeta {
            game: "minipente".to_string(),
            net: NetConfig {
                channels: 64,
                blocks: 6,
                value_hidden: 32,
            },
        };
        write_model_meta(&model, &meta).unwrap();
        assert_eq!(read_model_meta(&model).unwrap(), meta);
        // Flat JSON; every field is required.
        let json = fs::read_to_string(model_meta_path(&model)).unwrap();
        assert!(json.contains("\"game\": \"minipente\"") && json.contains("\"channels\": 64"));
        fs::write(
            model_meta_path(&model),
            r#"{"channels": 32, "blocks": 2, "value_hidden": 16}"#,
        )
        .unwrap();
        assert!(
            read_model_meta(&model).is_err(),
            "the game field is required"
        );
        fs::write(model_meta_path(&model), "not json").unwrap();
        assert!(read_model_meta(&model).is_err());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn check_game_rejects_a_model_for_another_game() {
        let meta = |game: &str| ModelMeta {
            game: game.to_string(),
            net: NetConfig {
                channels: 64,
                blocks: 6,
                value_hidden: 64,
            },
        };
        assert!(check_game::<MiniPenteState>(&meta("minipente")).is_ok());
        let err = check_game::<MiniPenteState>(&meta("connect4")).unwrap_err();
        assert!(err.to_string().contains("connect4"));
    }
}
