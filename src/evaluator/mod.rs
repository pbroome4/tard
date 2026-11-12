use burn::tensor::{Device, Tensor, TensorData};
use burn::module::Module;

use crate::gamestate::*;
use crate::tard::*;
use crate::utils::*;

pub mod batched;
pub mod replicated;

pub use batched::{BatchedBackend, BatchedClient};
pub use replicated::{ReplicatedBackend, ReplicatedClient};

////////////////////////////////////////////////////////////////////////
//
//                    Evaluator
//
///////////////////////////////////////////////////////////////////////
// Takes input state, evaluates it, and returns (policy, value) dual output.
// vector is a probability (NOT log) distribution over legal moves
// f32 is scalar score of current position.
pub trait Evaluator <G: GameState>: Send + 'static{
    fn evaluate (&mut self, g: &G) -> (Vec<f32>, f32);
}

////////////////////////////////////////////////////////////////////////
//
//                    EvalBackend
//
///////////////////////////////////////////////////////////////////////
// Hands out one Evaluator client per worker thread and propagates
// model updates from the training thread to those clients.
pub trait EvalBackend <G: GameState>: Send{
    type Client: Evaluator<G>;
    fn client(&mut self) -> Self::Client;
    fn update(&mut self, t: &Tard);
}

///////////////////////////////////////////////////////////////////////
//
//                      Local Evaluators
//
///////////////////////////////////////////////////////////////////////

// Always returns uniform probability over legal moves and 0.0 score.
#[derive(Clone)]
pub struct NullEvaluator{}

impl <G:GameState> Evaluator<G> for NullEvaluator{
    fn evaluate (&mut self, g: &G) -> (Vec<f32>, f32) {
        let p = vec![1.0; G::ACTN_SPC_SIZE];
        return (legalize_policy(p, g), 0.0);
    }
}


// Runs input state through Tard and returns dual head output
#[derive(Clone)]
pub struct TardEvaluator{
    t: Tard,
    device: Device, // where t lives; looked up once rather than per evaluation
}

impl TardEvaluator{
    pub fn new(t: &Tard)->Self{
        let t = t.valid();
        let device = t.devices()
                      .into_iter()
                      .next()
                      .expect("Expected at least one device");
        return Self {t, device};
    }
}

impl <G:GameState> Evaluator<G> for TardEvaluator{
    fn evaluate (&mut self, g: &G) -> (Vec<f32>, f32) {
        // A batch of 1
        let state_tens_data = TensorData::new(g.to_tensor(),
                                 [1, G::N_CHAN, G::ROWS, G::COLS]);
        let state_batch_tens = Tensor::<4>::from_data(state_tens_data, &self.device);
        let (p_batch, v_batch) = self.t.forward(state_batch_tens);
        let p_tens: Tensor<1> = p_batch.flatten(0, -1); // debatch
        let v_tens = v_batch.flatten::<1>(0, -1); // debatch
        let p: Vec<f32> = p_tens.exp().into_data().try_into_vec().unwrap();
        let v = v_tens.into_scalar::<f32>();
        let p = legalize_policy(p, g);
        return (p, v);
    }
}
