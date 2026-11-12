
use std::io::{self, Write}; // Bring the I/O traits into scope
use std::str::FromStr;
use tracing::Level;
use clap::Parser;

use pentard::pente::*;
use pentard::gamestate::*;
use pentard::utils::*;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args{
    /// trace, debug, info, warning, or error. Default info
    #[arg(short, long, default_value = "info")]
    log_level: String,
}


fn main(){
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_max_level(Level::from_str(&args.log_level).unwrap()) // Set max level: TRACE, DEBUG, INFO, WARN, ERROR, or OFF
        .init();

    let mut brd = PenteState::new();
    loop{
        println!("\n{}", brd);
        let player: &str = brd.active_plyr_str();
        print!("Enter move for {player} as row, col: ");
        io::stdout().flush().unwrap();
        match read_move(){
            Ok((row, col)) => {
                if !brd.play(row, col){
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