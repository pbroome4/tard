
use burn::tensor::{Device};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use std::io::{self, Write};
use std::ptr::read;

use pentard::evaluator::TardEvaluator;
use pentard::gamestate::GameState;
use pentard::pente::{PenteState, action_2_coord};
use pentard::utils::*;
use pentard::engine::*;

#[derive(Parser, Debug)]
#[command(about = "Play Pente against a trained Tard model")]
struct Args{
    /// Model file to play against
    #[arg(long, default_value = "model.bpk")]
    model: PathBuf,

    /// MCTS rollouts per computer move
    #[arg(long, default_value_t = 3000)]
    rollouts: usize,
}

fn main(){
    let args = Args::parse();
    let mut filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into());
    filter = filter.add_directive("pentard=info".parse().unwrap());
    filter = filter.add_directive("inference=info".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    info!("Starting inference");
    let device = Device::default();
    info!("Device: {:?}", &device);
    let tard = load_model(&args.model, &device).unwrap_or_else(|e|{
        panic!("Failed to load model {}: {:?}", args.model.display(), e)
    });
    let mut eval = TardEvaluator::new(&tard);

    let mut g = PenteState::new();
    print!("Is human first to play? (y/n) ");
    io::stdout().flush().unwrap();
    let Some(mut is_player_move) = read_yn() else{
        println!("\nInput closed, quitting");
        return;
    };
    println!("\n{}", g);
    loop{
        if is_player_move{
            let player: &str = g.active_plyr_str();
            print!("Enter move for {player} as row, col: ");
            io::stdout().flush().unwrap();
            let (row, col) = match read_move(){
                Ok(rc) => rc,
                Err(ReadError::Invalid(_)) => {
                    println!("Failed to read move");
                    continue;
                }
                Err(ReadError::Eof) => {
                    println!("\nInput closed, quitting");
                    return;
                }
            };
            if !g.play(row, col){
                println!("Illegal Move");
                continue;
            }
            if g.is_terminal(){
                break;
            }
            is_player_move = false;
            println!("\n{}", g);
        }
        if !is_player_move{
            let player: &str = g.active_plyr_str();
            println!("Computer thinking of move for {player}");
            let (action, val) = inference(&g, &mut eval, args.rollouts);
            let (row, col) = action_2_coord(action);
            assert!(g.play(row, col), "Engine chose an illegal move ({}, {})", row, col);
            println!("Computer played (row, col): ({}, {})", row, col);
            println!("Computer eval: {}", val);
            if g.is_terminal(){
                break;
            }
            is_player_move = true;
            println!("\n{}", g);
        }
    }
    println!("{}", g);
    println!("Game Over!");
}