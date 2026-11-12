
use std::io::{self, Write}; // Bring the I/O traits into scope
use std::str::FromStr;
use tracing::Level;
use clap::{Parser, ValueEnum};

use tard::minipente::MiniPenteState;
use tard::pente::PenteState;
use tard::hive::HiveState;
use tard::gamestate::*;
use tard::utils::*;


#[derive(Copy, Clone, Debug, ValueEnum)]
enum Game{
    /// Pente on a 9x9 board, no opening restrictions
    #[value(name = "minipente")]
    MiniPente,
    /// Official Pente: 19x19, White first, tournament opening rule
    Pente,
    /// Hive, base game (no expansion pieces)
    Hive,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Human-vs-human MiniPente, Pente or Hive in the terminal", long_about = None)]
struct Args{
    /// Which game to play
    #[arg(long, value_enum, default_value_t = Game::MiniPente)]
    game: Game,

    /// trace, debug, info, warning, or error. Default info
    #[arg(short, long, default_value = "info")]
    log_level: String,
}


fn main(){
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_max_level(Level::from_str(&args.log_level).unwrap()) // Set max level: TRACE, DEBUG, INFO, WARN, ERROR, or OFF
        .init();
    match args.game{
        Game::MiniPente => play::<MiniPenteState>(),
        Game::Pente => play::<PenteState>(),
        Game::Hive => play::<HiveState>(),
    }
}

fn play<G: TextGame>(){
    let mut brd = G::default();
    loop{
        println!("\n{}", brd);
        let player: &str = brd.active_player();
        print!("Enter move for {player} ({}): ", G::MOVE_FORMAT);
        io::stdout().flush().unwrap();
        let Some(line) = read_line() else{
            println!("\nInput closed, quitting");
            return;
        };
        match brd.parse_action(&line){
            Ok(action) => {
                brd.apply_action(action);
                if brd.is_terminal(){
                    break;
                }
            }
            Err(e) => println!("Illegal move: {}", e),
        }
    }
    println!("{}", brd);
    println!("Game Over!");
}
