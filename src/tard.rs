use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::activation::{relu, tanh, log_softmax};
use tracing::{trace, debug, info};
use core::fmt;
use bounded_vec_deque::BoundedVecDeque;
use rand::RngExt;

use crate::tard;


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
    pub fn new(in_chan: usize,
               rows: usize,
               cols: usize,
               action_space_size:usize,
               device: &Device) -> Self
    {
        info!("Creating Tard in_chan:{}, rows: {},  cols:{}", in_chan, rows, cols);
        let conv1 = Conv2dConfig::new([in_chan, 128], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(128).init(device);
        let mut resnets = Vec::new();
        for i in 0..10{ // N resnet layers
            resnets.push(ResNet::new(128, device));
        }

        let policy_conv = Conv2dConfig::new([128, 2], [1, 1]).init(device);
        let policy_bn = BatchNormConfig::new(2).init(device);
        let policy_fc = LinearConfig::new(2 * rows * cols, rows * cols).init(device);

        let value_conv = Conv2dConfig::new([128, 1], [1, 1]).init(device);
        let value_bn = BatchNormConfig::new(1).init(device);
        let value_fc1 = LinearConfig::new(action_space_size, 128).init(device);
        let value_fc2 = LinearConfig::new(128, 1).init(device);

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
            conv1: conv1,
            conv2: conv2,
            bn1: bn1,
            bn2:bn2
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
        let out = relu(out.clone() + identity);
        return out;
    }
}

////// Training structures and Batch Utilities ////
#[derive(Clone, Debug)]
pub struct TrainingElement{
    pub state: TensorData,       // rank 3       
    pub mcts_posterior: TensorData,  // rank 1
    pub outcome: TensorData,         // rank 1
}

pub struct TrainingBatch{
    pub state: Tensor<4>,
    pub mcts_posterior:Tensor<2>,
    pub outcome: Tensor<2>
}

impl TrainingBatch{
    pub fn from_training_elements(v: &Vec<TrainingElement>, device: &Device) -> TrainingBatch{
        let mut states = Vec::with_capacity(v.len());
        let mut mcts_posteriors = Vec::with_capacity(v.len());
        let mut outcomes = Vec::with_capacity(v.len());
        for te in v.into_iter(){
            let state_tens = Tensor::<3>::from_data(te.state.clone(), device);
            let mcts_tens = Tensor::<1>::from_data(te.mcts_posterior.clone(), device);
            let outcome_tens = Tensor::<1>::from_data(te.outcome.clone(), device);
            states.push(state_tens);
            mcts_posteriors.push(mcts_tens);
            outcomes.push(outcome_tens);
        }
        let state_batch = Tensor::stack(states, 0);
        let mcts_posterior_batch = Tensor::stack(mcts_posteriors, 0);
        let outcome_batch = Tensor::stack(outcomes, 0);
        let s = Self{
                state:state_batch,
                mcts_posterior:mcts_posterior_batch,
                outcome:outcome_batch
            };
        return s;
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

impl Default for TardLoss{
    fn default() -> Self {
        return Self {
            loss: Tensor::from_floats([0.0], &Default::default()),
            cross_entropy_loss: Tensor::from_floats([0.0], &Default::default()),
            mse_loss: Tensor::from_floats([0.0], &Default::default())
        }
    }
}

pub fn compute_loss(t: &Tard, v: &Vec<TrainingElement>) -> TardLoss{
    let device = t.devices().iter().next().unwrap().clone();
    let mut batch = TrainingBatch::from_training_elements(v, &device);
    //batch.state = batch.state.autodiff().require_grad();
    //batch.mcts_posterior = batch.mcts_posterior.autodiff().require_grad();
    //batch.outcome = batch.outcome.autodiff().require_grad();
    let (tard_policy_batch, tard_value_batch) = t.forward(batch.state);
    
    // policy networkloss
    debug!("mcts_posterior_logits: {}", batch.mcts_posterior);
    debug!("tard_policy_logits: {}", tard_policy_batch);
    let mcts_posterior_probs = batch.mcts_posterior.exp();
    debug_assert!(mcts_posterior_probs.shape() == tard_policy_batch.shape());
    let cross_entropy_terms = mcts_posterior_probs * tard_policy_batch;
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

    pub fn sample(&self, size: usize) -> Vec<TrainingElement> {
        let mut v = Vec::<TrainingElement>::new();
        let l = self.memory_.len();
        if l > 0{
            for _ in 0..size {
                let i = rand::rng().random_range(0..l);                
                v.push(self.memory_[i].clone());
            }
        }
        return v;
    }

    pub fn extend_front<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = TrainingElement>,
    {
        return self.memory_.extend(iter);
    }

    pub fn len(&self) -> usize{
        return self.memory_.len();
    }
}




