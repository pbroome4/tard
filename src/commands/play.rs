//! `tard play`: play a game in the terminal, against a trained model
//! (`--model`, whose sidecar names the game) or human vs human (`--game`).
use std::io::{self, Write};
use std::path::PathBuf;
use tracing::info;

use tard::evaluator::TardEvaluator;
use tard::games::{Game, TextGame};
use tard::mcts::best_move;
use tard::model::load_model;
use tard::with_game;

use super::game_of_model;

#[derive(clap::Args, Debug)]
pub struct Args {
    /// Model to play against (its <name>.net.json sidecar, which names the
    /// game, must sit next to it). Without one, two humans play.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Game for human vs human [default: minipente]. With --model the game
    /// comes from the model.
    #[arg(long, value_enum, conflicts_with = "model")]
    game: Option<Game>,

    /// MCTS rollouts per computer move
    #[arg(long, default_value_t = 3000)]
    rollouts: usize,
}

pub fn run(args: Args) {
    match &args.model {
        Some(model) => {
            with_game!(game_of_model(model), G => vs_computer::<G>(model, args.rollouts))
        }
        None => with_game!(args.game.unwrap_or(Game::MiniPente), G => vs_human::<G>()),
    }
}

fn vs_computer<G: TextGame>(model: &PathBuf, rollouts: usize) {
    info!("Playing {} against {}", G::NAME, model.display());
    let device = tard::device::pick_device();
    let (tard, net) = load_model::<G>(model, &device)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model.display(), e));
    info!("Network: {:?}", net);
    let mut eval = TardEvaluator::new(&tard);

    let mut g = G::default();
    print!("Is human first to play? (y/n) ");
    io::stdout().flush().unwrap();
    let Some(mut is_player_move) = read_yn() else {
        println!("\nInput closed, quitting");
        return;
    };
    println!("\n{}", g);
    while !g.is_terminal() {
        if is_player_move {
            let Some(action) = read_move(&g) else {
                println!("\nInput closed, quitting");
                return;
            };
            g.apply_action(action);
        } else {
            println!("Computer thinking of move for {}", g.active_player());
            let (action, val) = best_move(&g, &mut eval, rollouts);
            let text = g.action_text(action);
            g.apply_action(action);
            println!("Computer played: {}", text);
            println!("Computer eval: {}", val);
        }
        is_player_move = !is_player_move;
        println!("\n{}", g);
    }
    println!("Game Over!");
}

fn vs_human<G: TextGame>() {
    let mut g = G::default();
    println!("\n{}", g);
    while !g.is_terminal() {
        let Some(action) = read_move(&g) else {
            println!("\nInput closed, quitting");
            return;
        };
        g.apply_action(action);
        println!("\n{}", g);
    }
    println!("Game Over!");
}

// Prompts the player to move until they type a legal move. None on EOF.
fn read_move<G: TextGame>(g: &G) -> Option<usize> {
    loop {
        print!(
            "Enter move for {} ({}): ",
            g.active_player(),
            G::MOVE_FORMAT
        );
        io::stdout().flush().unwrap();
        match g.parse_action(&read_line()?) {
            Ok(action) => return Some(action),
            Err(e) => println!("Illegal move: {}", e),
        }
    }
}

// Reads one line from stdin. None on EOF or read error.
fn read_line() -> Option<String> {
    let mut input_buffer = String::new();
    return match io::stdin().read_line(&mut input_buffer) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(input_buffer),
    };
}

// Reads a y/n answer, re-prompting on anything else. None on EOF.
fn read_yn() -> Option<bool> {
    loop {
        let line = read_line()?;
        match line.trim().to_lowercase().as_str() {
            "y" | "yes" => return Some(true),
            "n" | "no" => return Some(false),
            _ => {
                print!("Please answer y or n: ");
                let _ = io::stdout().flush();
            }
        }
    }
}
