use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::activation::{relu, tanh, log_softmax};
use tracing::{debug, info};
use core::fmt;
use bounded_vec_deque::BoundedVecDeque;
use rand::RngExt;

use serde::{Deserialize, Serialize};

use crate::gamestate::GameState;

/// Size of the network's trunk and value head. The input and output layers
/// are sized from the GameState instead. A checkpoint only loads into a
/// network with the same NetConfig, so it is saved next to every model file
/// (see utils::save_model / utils::load_model).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetConfig{
    pub channels: usize,     // channels in the conv trunk
    pub blocks: usize,       // residual blocks in the trunk
    pub value_hidden: usize, // hidden units in the value head
}


#[derive(Debug, Module)]
pub struct Tard{
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
    pub fn for_game<G: GameState>(net: &NetConfig, device: &Device) -> Self{
        return Self::new(G::N_CHAN, G::ROWS, G::COLS, G::ACTN_SPC_SIZE, net, device);
    }

    pub fn new(in_chan: usize,
               rows: usize,
               cols: usize,
               action_space_size:usize,
               net: &NetConfig,
               device: &Device) -> Self
    {
        info!("Creating Tard in_chan:{}, rows: {},  cols:{}, {:?}", in_chan, rows, cols, net);
        let ch = net.channels;
        let conv1 = Conv2dConfig::new([in_chan, ch], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(ch).init(device);
        let mut resnets = Vec::new();
        for _ in 0..net.blocks{
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
        for resnet in self.resnets.iter(){
            x = resnet.forward(x);
        }

        // 2. Policy Head
        let p = self.policy_conv.forward(x.clone());
        let p = self.policy_bn.forward(p);
        let p = relu(p);
        let [batch_size, channels, height, width] = p.shape().dims();
        let p = p.reshape([batch_size, channels * height * width]);
        let p = self.policy_fc.forward(p);
        let p = log_softmax(p, 1);

        // 3. Value Head
        let v = self.value_conv.forward(x);
        let v = self.value_bn.forward(v);
        let v = relu(v);
        let [batch_size, channels, height, width] = v.shape().dims();
        let v = v.reshape([batch_size, channels * height * width]);
        let v = self.value_fc1.forward(v);
        let v = relu(v);
        let v = self.value_fc2.forward(v);
        let v = tanh(v);

        (p, v)
    }
}

#[derive(Debug, Module)]
struct ResNet{
    conv1: Conv2d,
    conv2: Conv2d,
    bn1: BatchNorm,
    bn2: BatchNorm,
}

impl ResNet{
    fn new(num_chan:usize, device: &Device) -> Self{
         let conv1 = Conv2dConfig::new([num_chan, num_chan], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([num_chan, num_chan], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        
        let bn1 = BatchNormConfig::new(num_chan).init(device);
        let bn2 = BatchNormConfig::new(num_chan).init(device);
        let s = Self{
            conv1,
            conv2,
            bn1,
            bn2
        };
        return s;
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4>{
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

////// Training structures and Batch Utilities ////
#[derive(Clone, Debug)]
pub struct TrainingElement{
    pub state: Vec<f32>,          // flattened N_CHAN x ROWS x COLS (GameState::to_tensor)
    pub mcts_posterior: Vec<f32>, // MCTS visit distribution, length ACTN_SPC_SIZE
    pub outcome: f32,             // game result from the side-to-move's perspective
}

pub struct TrainingBatch{
    pub state: Tensor<4>,
    pub mcts_posterior:Tensor<2>,
    pub outcome: Tensor<2>
}

impl TrainingBatch{
    // Packs the elements into one flat buffer per field, so each field is a
    // single host-to-device copy.
    pub fn from_training_elements<G: GameState>(v: &[&TrainingElement], device: &Device) -> TrainingBatch{
        let n = v.len();
        let mut states = Vec::with_capacity(n * G::N_CHAN * G::ROWS * G::COLS);
        let mut mcts_posteriors = Vec::with_capacity(n * G::ACTN_SPC_SIZE);
        let mut outcomes = Vec::with_capacity(n);
        for te in v{
            states.extend_from_slice(&te.state);
            mcts_posteriors.extend_from_slice(&te.mcts_posterior);
            outcomes.push(te.outcome);
        }
        return Self{
            state: Tensor::<4>::from_data(
                TensorData::new(states, [n, G::N_CHAN, G::ROWS, G::COLS]), device),
            mcts_posterior: Tensor::<2>::from_data(
                TensorData::new(mcts_posteriors, [n, G::ACTN_SPC_SIZE]), device),
            outcome: Tensor::<2>::from_data(
                TensorData::new(outcomes, [n, 1]), device),
        };
    }
}

impl fmt::Display for TrainingBatch{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.outcome)?;
        return Ok(());
    }
}


#[derive(Debug)]
pub struct TardLoss{
    pub loss: Tensor<1>,
    pub cross_entropy_loss: Tensor<1>,
    pub mse_loss: Tensor<1>
}

impl std::fmt::Display for TardLoss{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Loss: {}", self.loss.to_data())?;
        writeln!(f, "\tcross_entropy_loss: {}", self.cross_entropy_loss.to_data())?;
        writeln!(f, "\tmse_loss: {}", self.mse_loss.to_data())?;
        return Ok(());
    }
}

pub fn compute_loss<G: GameState>(t: &Tard, v: &[&TrainingElement]) -> TardLoss{
    let device = t.devices().first().unwrap().clone();
    let batch = TrainingBatch::from_training_elements::<G>(v, &device);
    let (tard_policy_batch, tard_value_batch) = t.forward(batch.state);
    
    // policy network loss: cross-entropy of the policy's log-probs against
    // the MCTS posterior probabilities
    debug!("mcts_posterior: {}", batch.mcts_posterior);
    debug!("tard_policy_log_probs: {}", tard_policy_batch);
    debug_assert!(batch.mcts_posterior.shape() == tard_policy_batch.shape());
    let cross_entropy_terms = batch.mcts_posterior * tard_policy_batch;
    let cross_entropy_loss = cross_entropy_terms.sum_dim(1)
                                                .mean()
                                                .neg();

    // value network loss
    debug!("batch_outcome: {}", batch.outcome);
    debug!("tard_value_batch: {}", tard_value_batch);
    debug_assert!(tard_value_batch.shape() == batch.outcome.shape());
    let mse_loss = (tard_value_batch - batch.outcome).powf_scalar(2.0)
                                                     .mean();
    
    let loss = cross_entropy_loss.clone() + mse_loss.clone();
    return TardLoss{loss, cross_entropy_loss, mse_loss};
}


pub struct TrainingBuffer{
    memory_: BoundedVecDeque<TrainingElement>,
}

impl TrainingBuffer{
    pub fn new(cap: usize) -> Self {
        return Self{
            memory_: BoundedVecDeque::with_capacity(cap, cap),
        };
    }

    // Samples `size` elements uniformly with replacement.
    // Returns an empty Vec if the buffer is empty.
    pub fn sample(&self, size: usize) -> Vec<&TrainingElement> {
        let mut v = Vec::with_capacity(size);
        let l = self.memory_.len();
        if l > 0{
            let mut rng = rand::rng();
            for _ in 0..size {
                v.push(&self.memory_[rng.random_range(0..l)]);
            }
        }
        return v;
    }

    // Appends elements at the back, evicting the oldest from the front
    // once the buffer is at capacity.
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = TrainingElement>,
    {
        return self.memory_.extend(iter);
    }

    pub fn len(&self) -> usize{
        return self.memory_.len();
    }

    pub fn is_empty(&self) -> bool{
        return self.memory_.is_empty();
    }
}






#[cfg(test)]
mod tests{
    use super::*;

    fn element(outcome: f32) -> TrainingElement{
        return TrainingElement{ state: vec![], mcts_posterior: vec![], outcome };
    }

    #[test]
    fn training_buffer_evicts_the_oldest_elements_at_capacity(){
        let mut buf = TrainingBuffer::new(3);
        assert!(buf.is_empty());
        buf.extend((0..5).map(|i| element(i as f32)));
        assert_eq!(buf.len(), 3);
        let kept: Vec<f32> = buf.memory_.iter().map(|e| e.outcome).collect();
        assert_eq!(kept, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn training_buffer_samples_with_replacement(){
        let mut buf = TrainingBuffer::new(10);
        assert!(buf.sample(5).is_empty(), "sampling an empty buffer returns nothing");
        buf.extend((0..3).map(|i| element(i as f32)));
        let s = buf.sample(50);
        assert_eq!(s.len(), 50);
        assert!(s.iter().all(|e| (0.0..3.0).contains(&e.outcome)));
    }
}
