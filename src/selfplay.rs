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
use tracing::{debug, info};

use crate::engine::{run_self_play, DirichletHParams, SelfPlayElement};
use crate::evaluator::{Evaluator, EvalBackend, TardEvaluator};
use crate::gamestate::GameState;
use crate::tard::Tard;


pub trait SelfPlaySource<G: GameState>{
    // Blocks until one finished game is available. None means the source has shut down.
    fn next_game(&mut self) -> Option<Vec<SelfPlayElement<G>>>;
    fn update_model(&mut self, t: &Tard);
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
        self.eval = TardEvaluator::new(t);
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
        return self.r_games.as_ref()?.recv().ok();
    }

    fn update_model(&mut self, t: &Tard){
        self.backend.update(t);
    }
}

impl <G: GameState, B: EvalBackend<G>> Drop for WorkerPool<G, B>{
    // Closing the game channel makes each worker exit after its current game.
    // Workers are joined before the backend drops so their evaluators stay valid.
    fn drop(&mut self){
        info!("Shutting down worker pool");
        drop(self.r_games.take());
        for h in self.handles.drain(..){
            let _ = h.join();
        }
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
    thread::spawn(move ||{
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
    })
}
