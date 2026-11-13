//////////////////////////////////////////////////////////////////////////////
//
//                      Self-Play Sources
//
//////////////////////////////////////////////////////////////////////////////
// run_self_play plays one game of the network against itself. The training
// loop pulls finished games from a SelfPlaySource and pushes updated models
// back into it. Implementations:
//   SingleThreaded         - plays games inline on the calling thread.
//   WorkerPool<B>          - N worker threads, each with a client from an
//                            EvalBackend (BatchedBackend, ReplicatedBackend).

use std::thread::{self, JoinHandle};
use crossbeam::channel::{Receiver, Sender};
use tracing::{debug, error, info, trace, warn};

use crate::evaluator::{Evaluator, EvalBackend, EvalStatsSnapshot, TardEvaluator};
use crate::games::GameState;
use crate::mcts::{search, select_action, DirichletHParams, MctsTree, C_PUCT};
use crate::model::Tard;
use crate::util::panic_message;


//////////////////////////// One game ////////////////////////////

#[derive(Debug)]
pub struct SelfPlayElement<G:GameState>{
    pub g: G,
    pub mcts_post: Vec<f32>,
    pub outcome: f32
}

// Plays one game against itself. Each move searches until the root has
// num_sims visits; the subtree under the chosen move is kept, so its visits
// count toward the next move's num_sims.
pub fn run_self_play <G: GameState>(
    eval: &mut impl Evaluator<G>,
    num_sims: usize,
    dhp: &DirichletHParams,
    temp_moves: usize,
) -> Vec<SelfPlayElement<G>>{
    debug!("Running self play");
    let mut sp_vec = Vec::new();
    let mut g = G::default();
    let mut tree = MctsTree::new(C_PUCT);
    let mut move_n = 0;
    while !g.is_terminal(){
        debug!("Current State: \n{}", g);
        debug!("Computing self-play move#: {}", move_n);
        let (mcts_post, _) = search(&mut tree, &g, eval, num_sims, dhp);
        // Sample for the first temp_moves plies (opening variety), then play greedily.
        let action = select_action(&mcts_post, move_n >= temp_moves);
        let sp_element = SelfPlayElement{
            g: g.clone(),
            mcts_post,
            outcome: 0.0
        };
        sp_vec.push(sp_element);
        debug!("Playing action: {}", action);
        g.apply_action(action);
        tree = tree.reroot(action);
        move_n += 1;
    }
    debug!("Current State: \n{}", g);
    let outcome = g.get_terminal_value();
    // sp_vec has list of all non-terminal states.
    // outcome is relative to terminal state. So we iterate
    // in reverse and polarity starts at -1.0
    let mut polarity = -1.0;
    for spe in sp_vec.iter_mut().rev(){
        spe.outcome = polarity * outcome;
        polarity *= -1.0;
    }
    trace!("{:?}", &sp_vec);
    return sp_vec;
}


//////////////////////////// Sources ////////////////////////////

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
    use crate::games::minipente::MiniPenteState;
    use crate::games::pente::{self, PenteState};
    use crate::games::hive::{self, HiveState};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn noise() -> DirichletHParams{
        return DirichletHParams{ epsilon: 0.25, alpha: 0.25 };
    }

    fn assert_distribution(p: &[f32]){
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sums to {sum}");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn self_play_outcomes_alternate_from_the_winners_perspective(){
        for _ in 0..3{
            let game = run_self_play::<MiniPenteState>(&mut NullEvaluator{}, 16, &noise(), 4);
            assert!(!game.is_empty());
            assert!(!game[0].g.is_terminal());
            let last = game.last().unwrap().outcome;
            // The last mover either won (+1) or the game was drawn (0).
            assert!(last == 1.0 || last == 0.0, "last outcome {last}");
            for pair in game.windows(2){
                assert_eq!(pair[0].outcome, -pair[1].outcome);
            }
            for spe in &game{
                assert_distribution(&spe.mcts_post);
            }
        }
    }

    #[test]
    fn pente_self_play_follows_the_tournament_rule(){
        // Root noise is skipped for the forced first move (one legal child).
        let game = run_self_play::<PenteState>(&mut NullEvaluator{}, 8, &noise(), 4);
        assert!(game.len() >= 3);
        assert_eq!(game[0].mcts_post[pente::coord_2_action(9, 9)], 1.0,
                   "the first stone must go on the center");
        for r in 7..=11{
            for c in 7..=11{
                assert_eq!(game[2].mcts_post[pente::coord_2_action(r, c)], 0.0,
                           "first player's second stone inside the central 5x5 at ({r}, {c})");
            }
        }
        for spe in &game{
            assert_distribution(&spe.mcts_post);
        }
    }

    #[test]
    fn hive_self_play_runs_to_the_end(){
        let game = run_self_play::<HiveState>(&mut NullEvaluator{}, 4, &noise(), 4);
        assert!(!game.is_empty() && game.len() <= hive::MAX_PLIES);
        for pair in game.windows(2){
            assert_eq!(pair[0].outcome, -pair[1].outcome);
        }
        for spe in &game{
            assert_distribution(&spe.mcts_post);
        }
    }

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
