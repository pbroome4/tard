// Plays two trained models against each other and reports the match result.
use burn::tensor::Device;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tracing::info;
use tracing_subscriber::filter::{LevelFilter, EnvFilter};

use pentard::engine::{mcts, select_action, DirichletHParams};
use pentard::evaluator::TardEvaluator;
use pentard::gamestate::GameState;
use pentard::pente::{PenteState, action_2_coord};
use pentard::tard::Tard;
use pentard::utils::load_model;

#[derive(Parser, Debug)]
#[command(about = "Play two trained Tard models against each other")]
struct Args{
    /// First model file (its <name>.net.json sidecar must sit next to it)
    #[arg(long)]
    model_a: PathBuf,

    /// Second model file
    #[arg(long)]
    model_b: PathBuf,

    /// Number of games. Colors alternate: A is Black in even-numbered games.
    #[arg(long, default_value_t = 20)]
    games: usize,

    /// MCTS rollouts per move for both models
    #[arg(long, default_value_t = 800)]
    rollouts: usize,

    /// Rollouts for model A only (overrides --rollouts)
    #[arg(long)]
    rollouts_a: Option<usize>,

    /// Rollouts for model B only (overrides --rollouts)
    #[arg(long)]
    rollouts_b: Option<usize>,

    /// Sample the first N plies from the MCTS visit distribution instead of
    /// taking the best move, so games differ. With 0 every game with the
    /// same colors is identical, since both models play deterministically.
    #[arg(long, default_value_t = 4)]
    temp_moves: usize,

    /// Games played at the same time, each on its own thread
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Print the final board of every game
    #[arg(long)]
    show: bool,

    /// Watch games live: print the board and each model's evaluation after
    /// every move. Plays one game at a time (ignores --threads).
    #[arg(long)]
    watch: bool,

    /// With --watch, pause this many milliseconds after each move
    #[arg(long, default_value_t = 0)]
    delay_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome{ AWins, BWins, Draw }

struct Player{
    eval: TardEvaluator,
    rollouts: usize,
}

// How play_game reports moves as they happen.
struct Watch{
    game: usize,
    delay: Duration,
}

// Plays one game. Returns the result and the number of plies.
// With `watch`, prints the board after every move.
fn play_game(a: &mut Player, b: &mut Player, a_is_black: bool, temp_moves: usize,
             watch: Option<&Watch>)
    -> (Outcome, usize, PenteState)
{
    let dhp = DirichletHParams{ epsilon: 0.0, ..Default::default() };
    let mut g = PenteState::default();
    let mut ply = 0;
    // Black moves on even plies.
    let a_to_move = |ply: usize| (ply % 2 == 0) == a_is_black;
    if let Some(w) = watch{
        let colors = if a_is_black { "A=Black B=White" } else { "A=White B=Black" };
        println!("===== game {} ({}) =====\n{}", w.game, colors, g);
    }
    while !g.is_terminal(){
        let a_moves = a_to_move(ply);
        let p = if a_moves { &mut *a } else { &mut *b };
        let mover = format!("{} ({})", if a_moves { "A" } else { "B" }, g.active_plyr_str());
        let (post, val) = mcts(&g, &mut p.eval, p.rollouts, &dhp);
        let sampled = ply < temp_moves;
        let action = select_action(&post, !sampled);
        g.apply_action(action);
        ply += 1;
        if let Some(w) = watch{
            let (row, col) = action_2_coord(action);
            println!("ply {}: {} plays ({}, {}){}  eval for {}: {:+.2}  visits {:.0}%",
                     ply, mover, row, col, if sampled { " [sampled]" } else { "" },
                     mover, val, 100.0 * post[action]);
            println!("{}", g);
            if !w.delay.is_zero(){
                std::thread::sleep(w.delay);
            }
        }
    }
    // The terminal value is from the side to move's perspective:
    // -1 means the player who made the last move won.
    let result = if g.get_terminal_value() == 0.0{
        Outcome::Draw
    }else if a_to_move(ply - 1){
        Outcome::AWins
    }else{
        Outcome::BWins
    };
    return (result, ply, g);
}

fn load(p: &PathBuf, device: &Device) -> Tard{
    let (t, net) = load_model::<PenteState>(p, device).unwrap_or_else(|e|{
        panic!("Failed to load model {}: {}", p.display(), e)
    });
    info!("{}: {:?}", p.display(), net);
    return t;
}

// Elo difference implied by a score fraction s in (0, 1).
fn elo_diff(s: f64) -> f64{
    return -400.0 * (1.0 / s - 1.0).log10() + 0.0; // + 0.0 turns -0 into 0
}

fn main(){
    let args = Args::parse();
    let filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into())
        .add_directive("pentard=info".parse().unwrap())
        .add_directive("arena=info".parse().unwrap());
    tracing_subscriber::fmt().with_env_filter(filter).init();
    assert!(args.games > 0, "--games must be at least 1");
    assert!(args.threads > 0, "--threads must be at least 1");

    let device = Device::default();
    let tard_a = load(&args.model_a, &device);
    let tard_b = load(&args.model_b, &device);
    let rollouts_a = args.rollouts_a.unwrap_or(args.rollouts);
    let rollouts_b = args.rollouts_b.unwrap_or(args.rollouts);
    println!("A: {} ({} rollouts)", args.model_a.display(), rollouts_a);
    println!("B: {} ({} rollouts)", args.model_b.display(), rollouts_b);
    // Watching interleaves boards from concurrent games, so play one at a time.
    let threads = if args.watch { 1 } else { args.threads };
    println!("{} games, first {} plies sampled, {} threads\n",
             args.games, args.temp_moves, threads);

    let next_game = AtomicUsize::new(0);
    // (game index, a_is_black, result, plies)
    let results = Mutex::new(Vec::new());
    std::thread::scope(|s|{
        for _ in 0..threads.min(args.games){
            let mut a = Player{ eval: TardEvaluator::new(&tard_a), rollouts: rollouts_a };
            let mut b = Player{ eval: TardEvaluator::new(&tard_b), rollouts: rollouts_b };
            let (next_game, results, args) = (&next_game, &results, &args);
            s.spawn(move ||{
                loop{
                    let i = next_game.fetch_add(1, Ordering::Relaxed);
                    if i >= args.games{
                        break;
                    }
                    let a_is_black = i % 2 == 0;
                    let watch = Watch{ game: i, delay: Duration::from_millis(args.delay_ms) };
                    let watch = args.watch.then_some(&watch);
                    let (r, plies, g) = play_game(&mut a, &mut b, a_is_black, args.temp_moves, watch);
                    let colors = if a_is_black { "A=Black B=White" } else { "A=White B=Black" };
                    let outcome = match r{
                        Outcome::AWins => "A wins",
                        Outcome::BWins => "B wins",
                        Outcome::Draw => "draw",
                    };
                    let mut res = results.lock().unwrap();
                    res.push((i, a_is_black, r, plies));
                    println!("game {:>3} ({}): {:<6} in {} plies  [{}/{} done]",
                             i, colors, outcome, plies, res.len(), args.games);
                    if args.show && !args.watch{
                        println!("{}", g);
                    }
                }
            });
        }
    });

    let res = results.into_inner().unwrap();
    let count = |f: &dyn Fn(&(usize, bool, Outcome, usize)) -> bool| res.iter().filter(|x| f(x)).count();
    let a_wins = count(&|x| x.2 == Outcome::AWins);
    let b_wins = count(&|x| x.2 == Outcome::BWins);
    let draws = count(&|x| x.2 == Outcome::Draw);
    let a_black_wins = count(&|x| x.1 && x.2 == Outcome::AWins);
    let a_white_wins = count(&|x| !x.1 && x.2 == Outcome::AWins);
    let a_black_games = count(&|x| x.1);
    let a_white_games = res.len() - a_black_games;
    let black_wins = count(&|x| (x.1 && x.2 == Outcome::AWins) || (!x.1 && x.2 == Outcome::BWins));
    let mean_plies = res.iter().map(|x| x.3).sum::<usize>() as f64 / res.len() as f64;

    let n = res.len() as f64;
    let score = (a_wins as f64 + 0.5 * draws as f64) / n;
    println!("\nA {} - B {} - draws {}  (A scores {:.1}%)", a_wins, b_wins, draws, 100.0 * score);
    println!("A won {}/{} as Black, {}/{} as White", a_black_wins, a_black_games, a_white_wins, a_white_games);
    println!("Black won {}/{} games; mean game length {:.1} plies", black_wins, res.len(), mean_plies);
    if score > 0.0 && score < 1.0{
        // Standard error of the score, from the per-game results.
        let var = res.iter().map(|x|{
            let s = match x.2 { Outcome::AWins => 1.0, Outcome::BWins => 0.0, Outcome::Draw => 0.5 };
            (s - score) * (s - score)
        }).sum::<f64>() / n;
        let se = (var / n).sqrt();
        let lo = (score - 1.96 * se).max(1e-3);
        let hi = (score + 1.96 * se).min(1.0 - 1e-3);
        println!("Elo difference (A - B): {:+.0}  (95% range roughly {:+.0} to {:+.0})",
                 elo_diff(score), elo_diff(lo), elo_diff(hi));
    }else{
        println!("Elo difference: unbounded (one side won every game); play more games to estimate it");
    }
}
