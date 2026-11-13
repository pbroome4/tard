use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use burn::tensor::{Device, Tensor, TensorData};
use burn::module::Module;

use crate::games::*;
use crate::model::Tard;
use crate::util::normalize;
use tracing::warn;

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
    // Cumulative network-evaluation counters for everything this backend
    // runs. Backends that don't run a network can keep the zero default.
    fn stats(&self) -> EvalStatsSnapshot{
        return EvalStatsSnapshot::default();
    }
}

////////////////////////////////////////////////////////////////////////
//
//                    EvalStats
//
///////////////////////////////////////////////////////////////////////
// Throughput counters for network evaluations, shared (via Arc) by every
// thread that runs forward passes for one self-play source. Updates are
// relaxed atomics: cheap, and exact once read after the fact.
#[derive(Clone, Default)]
pub struct EvalStats(Arc<EvalCounters>);

#[derive(Default)]
struct EvalCounters{
    positions: AtomicU64, // positions evaluated
    batches: AtomicU64,   // forward passes
    busy_ns: AtomicU64,   // time inside forward passes (incl. upload and readback)
}

// A point-in-time copy of EvalStats; subtract two with `since` for a window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvalStatsSnapshot{
    pub positions: u64,
    pub batches: u64,
    pub busy: Duration,
}

impl EvalStats{
    // Records one forward pass over `positions` positions that took `busy`.
    pub fn record(&self, positions: usize, busy: Duration){
        self.0.positions.fetch_add(positions as u64, Ordering::Relaxed);
        self.0.batches.fetch_add(1, Ordering::Relaxed);
        self.0.busy_ns.fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> EvalStatsSnapshot{
        return EvalStatsSnapshot{
            positions: self.0.positions.load(Ordering::Relaxed),
            batches: self.0.batches.load(Ordering::Relaxed),
            busy: Duration::from_nanos(self.0.busy_ns.load(Ordering::Relaxed)),
        };
    }
}

impl EvalStatsSnapshot{
    // The counts accumulated between `earlier` and self.
    pub fn since(&self, earlier: &EvalStatsSnapshot) -> EvalStatsSnapshot{
        return EvalStatsSnapshot{
            positions: self.positions.saturating_sub(earlier.positions),
            batches: self.batches.saturating_sub(earlier.batches),
            busy: self.busy.saturating_sub(earlier.busy),
        };
    }
}

pub fn legalize_policy(mut p: Vec<f32>, g: &impl GameState) -> Vec<f32>{
    let legal_mask = g.legal_actions_mask();
    for i in 0..p.len(){
        if !legal_mask[i]{
            p[i] = 0.0;
        }
    }
    let sum: f32 = p.iter().sum();
    if !(sum > 0.0 && sum.is_finite()){
        // All legal moves got zero (underflow) or non-finite (NaN/inf)
        // probability. Fall back to uniform over legal moves rather than
        // producing NaNs.
        warn!("Policy has no usable mass over legal moves (sum = {}); using uniform", sum);
        p = legal_mask.iter().map(|&legal| if legal {1.0} else {0.0}).collect();
    }
    return normalize(p);
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
    stats: EvalStats,
}

impl TardEvaluator{
    pub fn new(t: &Tard)->Self{
        return Self::with_stats(t, EvalStats::default());
    }

    // Like new(), but records into existing (shared) counters.
    pub fn with_stats(t: &Tard, stats: EvalStats)->Self{
        let t = t.valid();
        let device = t.devices()
                      .into_iter()
                      .next()
                      .expect("Expected at least one device");
        return Self {t, device, stats};
    }

    pub fn stats(&self) -> &EvalStats{
        return &self.stats;
    }
}

impl <G:GameState> Evaluator<G> for TardEvaluator{
    fn evaluate (&mut self, g: &G) -> (Vec<f32>, f32) {
        // A batch of 1
        let state_tens_data = TensorData::new(g.to_tensor(),
                                 [1, G::N_CHAN, G::ROWS, G::COLS]);
        let start = Instant::now();
        let state_batch_tens = Tensor::<4>::from_data(state_tens_data, &self.device);
        let (p_batch, v_batch) = self.t.forward(state_batch_tens);
        let p_tens: Tensor<1> = p_batch.flatten(0, -1); // debatch
        let v_tens = v_batch.flatten::<1>(0, -1); // debatch
        let p: Vec<f32> = p_tens.exp().into_data().try_into_vec().unwrap();
        let v = v_tens.into_scalar::<f32>();
        self.stats.record(1, start.elapsed());
        let p = legalize_policy(p, g);
        return (p, v);
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use crate::games::minipente::MiniPenteState;

    #[test]
    fn legalize_policy_masks_illegal_moves_and_renormalizes(){
        let mut g = MiniPenteState::new();
        g.play(0, 0);
        let mut p = vec![0.0; 81];
        p[0] = 0.5; // occupied
        p[1] = 0.25;
        p[2] = 0.25;
        let p = legalize_policy(p, &g);
        assert_eq!(p[0], 0.0);
        assert!((p[1] - 0.5).abs() < 1e-6 && (p[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn legalize_policy_falls_back_to_uniform_without_usable_mass(){
        let mut g = MiniPenteState::new();
        g.play(4, 4);
        let all_on_illegal = {
            let mut p = vec![0.0; 81];
            p[40] = 1.0;
            p
        };
        for bad in [vec![0.0; 81], vec![f32::NAN; 81], vec![f32::INFINITY; 81], all_on_illegal]{
            let p = legalize_policy(bad, &g);
            assert_eq!(p[40], 0.0);
            for (a, &x) in p.iter().enumerate(){
                if a != 40{
                    assert!((x - 1.0 / 80.0).abs() < 1e-6, "action {a}: {x}");
                }
            }
        }
    }

    #[test]
    fn eval_stats_accumulate_across_clones_and_diff_by_window(){
        let stats = EvalStats::default();
        let shared = stats.clone();
        stats.record(100, Duration::from_millis(5));
        let before = stats.snapshot();
        shared.record(28, Duration::from_millis(2));
        shared.record(4, Duration::from_millis(1));
        let window = stats.snapshot().since(&before);
        assert_eq!(window, EvalStatsSnapshot{ positions: 32, batches: 2, busy: Duration::from_millis(3) });
        assert_eq!(stats.snapshot().positions, 132);
    }
}
