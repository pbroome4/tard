//! Packing training examples into a batch, and the AlphaZero loss.

use burn::prelude::*;
use burn::tensor::Device;
use core::fmt;
use tracing::debug;

use crate::games::GameState;
use crate::model::Tard;
use super::replay::TrainingElement;

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
