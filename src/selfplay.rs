//////////////////////////////////////////////////////////////////////////////
//
//                      Self-Play Sources
//
//////////////////////////////////////////////////////////////////////////////
// The training loop pulls finished self-play games from a SelfPlaySource and
// pushes updated models back into it. Implementations:
//   SingleThreaded         - plays games inline on the calling thread.
//   WorkerPool<B>          - N worker threads, each with a client from an
//                            EvalBackend (BatchedBackend, ReplicatedBackend).

use std::thread::{self, JoinHandle};
use crossbeam::channel::{Receiver, Sender};
use tracing::{debug, error, info, warn};

use crate::engine::{run_self_play, DirichletHParams, SelfPlayElement};
use crate::evaluator::{Evaluator, EvalBackend, EvalStatsSnapshot, TardEvaluator};
use crate::gamestate::GameState;
use crate::tard::Tard;
use crate::utils::panic_message;


pub trait SelfPlaySource<G: GameState>{
    // Blocks until one finished game is available. None means the source has shut down.
    fn next_game(&mut self) -> Option<Vec<SelfPlayElement<G>>>;
    fn update_model(&mut self, t: &Tard);
    // Cumulative network-evaluation counters for this source's self-play.
    fn eval_stats(&self) -> EvalStatsSnapshot;
}


//////////////////////////// Single threaded ////////////////////////////

pub struct SingleThreaded{
    eval: TardEvaluator,
    n_rollouts: usize,
    dhp: DirichletHParams,
    temp_moves: usize,
}

impl SingleThreaded{
    pub fn new(t: &Tard, n_rollouts: usize, dhp: DirichletHParams, temp_moves: usize) -> Self{
        return Self{
            eval: TardEvaluator::new(t),
            n_rollouts,
            dhp,
            temp_moves,
        };
    }
}

impl <G: GameState> SelfPlaySource<G> for SingleThreaded{
    fn next_game(&mut self) -> Option<Vec<SelfPlayElement<G>>>{
        return Some(run_self_play::<G>(&mut self.eval, self.n_rollouts, &self.dhp, self.temp_moves));
    }

    fn update_model(&mut self, t: &Tard){
        self.eval = TardEvaluator::with_stats(t, self.eval.stats().clone());
    }

    fn eval_stats(&self) -> EvalStatsSnapshot{
        return self.eval.stats().snapshot();
    }
}


//////////////////////////// Worker pool ////////////////////////////

pub struct WorkerPool<G: GameState, B: EvalBackend<G>>{
    backend: B,
    r_games: Option<Receiver<Vec<SelfPlayElement<G>>>>,
    handles: Vec<JoinHandle<()>>,
}

impl <G: GameState, B: EvalBackend<G>> WorkerPool<G, B>{
    pub fn new(
        mut backend: B,
        n_workers: usize,
        n_rollouts: usize,
        dhp: DirichletHParams,
        temp_moves: usize) -> Self
    {
        let (s_games, r_games) = crossbeam::channel::bounded(n_workers.max(1));
        let mut handles = Vec::with_capacity(n_workers);
        for i in 0..n_workers{
            let eval = backend.client();
            handles.push(start_worker(i, s_games.clone(), eval, n_rollouts, dhp.clone(), temp_moves));
        }
        return Self{
            backend,
            r_games: Some(r_games),
            handles,
        };
    }
}

impl <G: GameState, B: EvalBackend<G>> SelfPlaySource<G> for WorkerPool<G, B>{
    fn next_game(&mut self) -> Option<Vec<SelfPlayElement<G>>>{
        // Workers only exit on their own by panicking, so report any that have.
        self.reap_finished_workers();
        match self.r_games.as_ref()?.recv(){
            Ok(game) => Some(game),
            Err(_) => {
                // Every worker has dropped its sender, i.e. they all exited.
                error!("All self-play workers have exited");
                self.join_all_workers();
                None
            }
        }
    }

    fn update_model(&mut self, t: &Tard){
        self.backend.update(t);
    }

    fn eval_stats(&self) -> EvalStatsSnapshot{
        return self.backend.stats();
    }
}

impl <G: GameState, B: EvalBackend<G>> WorkerPool<G, B>{
    // Joins workers that have already exited and logs any panics.
    fn reap_finished_workers(&mut self){
        let before = self.handles.len();
        let mut i = 0;
        while i < self.handles.len(){
            if self.handles[i].is_finished(){
                join_and_report(self.handles.swap_remove(i));
            }else{
                i += 1;
            }
        }
        if self.handles.len() < before{
            warn!("{} self-play worker(s) exited; {} still running",
                  before - self.handles.len(), self.handles.len());
        }
    }

    fn join_all_workers(&mut self){
        for h in self.handles.drain(..){
            join_and_report(h);
        }
    }
}

// Joins a thread, logging its panic message if it panicked.
fn join_and_report(h: JoinHandle<()>){
    let name = h.thread().name().unwrap_or("<unnamed>").to_string();
    if let Err(payload) = h.join(){
        error!("Thread {} panicked: {}", name, panic_message(&payload));
    }
}

impl <G: GameState, B: EvalBackend<G>> Drop for WorkerPool<G, B>{
    // Closing the game channel makes each worker exit after its current game.
    // Workers are joined before the backend drops so their evaluators stay valid.
    fn drop(&mut self){
        info!("Shutting down worker pool");
        drop(self.r_games.take());
        self.join_all_workers();
        info!("Worker pool shut down");
    }
}

fn start_worker<G: GameState>(
    thread_id: usize,
    s: Sender<Vec<SelfPlayElement<G>>>,
    mut eval: impl Evaluator<G>,
    n_rollouts: usize,
    dhp: DirichletHParams,
    temp_moves: usize) -> JoinHandle<()>
{
    debug!("Spawning worker thread #{}", thread_id);
    thread::Builder::new().name(format!("selfplay-{}", thread_id)).spawn(move ||{
        let mut self_play_count = 0;
        loop{
            let game_data = run_self_play::<G>(&mut eval, n_rollouts, &dhp, temp_moves);
            debug!("Worker #{} self play game {} complete, {} training elements",
                   thread_id, self_play_count, game_data.len());
            if s.send(game_data).is_err(){
                debug!("Worker #{} game channel closed, shutting down", thread_id);
                break;
            }
            self_play_count += 1;
        }
    }).expect("Failed to spawn self-play worker thread")
}


#[cfg(test)]
mod tests{
    use super::*;
    use crate::evaluator::NullEvaluator;
    use crate::minipente::MiniPenteState;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Uniform evaluator that panics on its first call if `panics` is set.
    struct MaybePanic{ panics: bool }
    impl Evaluator<MiniPenteState> for MaybePanic{
        fn evaluate(&mut self, g: &MiniPenteState) -> (Vec<f32>, f32){
            if self.panics{
                panic!("simulated evaluator failure");
            }
            return NullEvaluator{}.evaluate(g);
        }
    }

    // Hands out clients; the first `n_panicking` of them panic.
    struct FakeBackend{ handed_out: Arc<AtomicUsize>, n_panicking: usize }
    impl EvalBackend<MiniPenteState> for FakeBackend{
        type Client = MaybePanic;
        fn client(&mut self) -> MaybePanic{
            let i = self.handed_out.fetch_add(1, Ordering::SeqCst);
            return MaybePanic{ panics: i < self.n_panicking };
        }
        fn update(&mut self, _t: &Tard){}
    }

    fn pool(n_workers: usize, n_panicking: usize) -> WorkerPool<MiniPenteState, FakeBackend>{
        let backend = FakeBackend{ handed_out: Arc::new(AtomicUsize::new(0)), n_panicking };
        return WorkerPool::new(backend, n_workers, 4, DirichletHParams::default(), 2);
    }

    #[test]
    fn worker_pool_produces_games_and_shuts_down(){
        let mut p = pool(2, 0);
        let game = SelfPlaySource::<MiniPenteState>::next_game(&mut p).expect("a game");
        assert!(!game.is_empty());
        drop(p); // must join the workers rather than hang
    }

    #[test]
    fn worker_pool_returns_none_once_every_worker_has_panicked(){
        let mut p = pool(3, 3);
        assert!(SelfPlaySource::<MiniPenteState>::next_game(&mut p).is_none());
        assert!(p.handles.is_empty(), "dead workers should have been joined");
    }

    #[test]
    fn worker_pool_reaps_a_panicked_worker_and_keeps_going(){
        let mut p = pool(3, 1);
        // The panicking worker dies on its first evaluation; it is reaped
        // on a later next_game call while the others keep producing games.
        for _ in 0..200{
            assert!(SelfPlaySource::<MiniPenteState>::next_game(&mut p).is_some());
            if p.handles.len() == 2{
                return;
            }
        }
        panic!("panicked worker was never reaped");
    }
}
