
use std::io::{self, Write}; // Bring the I/O traits into scope
use std::str::FromStr;
use tracing::Level;
use clap::{Parser, ValueEnum};

use tard::minipente::MiniPenteState;
use tard::pente::PenteState;
use tard::gamestate::*;
use tard::utils::*;


#[derive(Copy, Clone, Debug, ValueEnum)]
enum Game{
    /// Pente on a 9x9 board, no opening restrictions
    #[value(name = "minipente")]
    MiniPente,
    /// Official Pente: 19x19, White first, tournament opening rule
    Pente,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Human-vs-human MiniPente or Pente in the terminal", long_about = None)]
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
    }
}

fn play<G: GridGame>(){
    let mut brd = G::default();
    loop{
        println!("\n{}", brd);
        let player: &str = brd.active_player();
        print!("Enter move for {player} as row, col: ");
        io::stdout().flush().unwrap();
        match read_move(){
            Ok((row, col)) => {
                if !brd.play_at(row, col){
                    println!("Illegal Move");
                }else if brd.is_terminal(){
                    break;
                }
            }
            Err(ReadError::Invalid(_)) => println!("Failed to read move"),
            Err(ReadError::Eof) => {
                println!("\nInput closed, quitting");
                return;
            }
        }
    }
    println!("{}", brd);
    println!("Game Over!");
}
