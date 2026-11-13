////////////////////////////////////////////////////////////////////
//
//                      Replicated (Decentralized) Evaluator
//
////////////////////////////////////////////////////////////////////
// Each client owns a local copy of the model and evaluates positions one at
// a time. The backend publishes new weights into a shared, versioned slot;
// clients notice the version change on their next evaluation and copy the
// latest model.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use burn::module::Module;
use tracing::{debug, info};

use crate::games::*;
use crate::model::Tard;
use super::{Evaluator, EvalBackend, EvalStats, EvalStatsSnapshot, TardEvaluator};


struct Shared{
    version: AtomicU64,
    model: Mutex<Tard>,
}

impl Shared{
    // Returns (version, model) consistently: version is read under the lock,
    // and update() only bumps it while holding the lock.
    fn snapshot(&self) -> (u64, Tard){
        let m = self.model.lock().unwrap();
        let v = self.version.load(Ordering::Acquire);
        return (v, m.clone());
    }
}


pub struct ReplicatedClient{
    shared: Arc<Shared>,
    local: TardEvaluator,
    seen: u64,
}

impl <G:GameState> Evaluator<G> for ReplicatedClient{
    fn evaluate(&mut self, g: &G) -> (Vec<f32>, f32){
        if self.shared.version.load(Ordering::Acquire) != self.seen{
            let (v, t) = self.shared.snapshot();
            debug!("Replicated client updating model v{} -> v{}", self.seen, v);
            self.local = TardEvaluator::with_stats(&t, self.local.stats().clone());
            self.seen = v;
        }
        return self.local.evaluate(g);
    }
}


pub struct ReplicatedBackend<G>{
    shared: Arc<Shared>,
    stats: EvalStats, // shared by all clients
    _marker: PhantomData<fn() -> G>,
}

impl <G:GameState> ReplicatedBackend<G>{
    pub fn new(t: &Tard) -> Self{
        let shared = Shared{
            version: AtomicU64::new(0),
            model: Mutex::new(t.valid()),
        };
        return Self{
            shared: Arc::new(shared),
            stats: EvalStats::default(),
            _marker: PhantomData,
        };
    }
}

impl <G:GameState> EvalBackend<G> for ReplicatedBackend<G>{
    type Client = ReplicatedClient;

    fn client(&mut self) -> ReplicatedClient{
        let (v, t) = self.shared.snapshot();
        return ReplicatedClient{
            shared: self.shared.clone(),
            local: TardEvaluator::with_stats(&t, self.stats.clone()),
            seen: v,
        };
    }

    fn update(&mut self, t: &Tard){
        let mut m = self.shared.model.lock().unwrap();
        *m = t.valid();
        let v = self.shared.version.fetch_add(1, Ordering::AcqRel) + 1;
        info!("Published replicated model v{}", v);
    }

    fn stats(&self) -> EvalStatsSnapshot{
        return self.stats.snapshot();
    }
}
