
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
    info!("Is human first to play? (y/n)");
    let mut is_player_move = read_yn();
    info!("\n{}", g);
    loop{
        if is_player_move{
            let player: &str = g.active_plyr_str();
            info!("Enter move for {player} as row, col: ");
            io::stdout().flush().unwrap();
            let Ok((row, col)) = read_move() else{
                info!("Failed to read move");
                continue;
            };
            if !g.play(row, col){
                info!("Illegal Move");
                continue;
            }
            if g.is_terminal(){
                break;
            }
            is_player_move = false;
            info!("\n{}", g);
        }
        if !is_player_move{
            let player: &str = g.active_plyr_str();
            info!("Computer thinking of move for {player}");
            let (action, val) = inference(&g, &mut eval, 3000);
            let (row, col) = action_2_coord(action);
            g.play(row, col);
            info!("Computer played (row, col): ({}, {})", row, col);
            info!("Computer eval: {}", val);
            if g.is_terminal(){
                break;
            }
            is_player_move = true;
            info!("\n{}", g);
        }
    }
    info!("{}", g);
    info!("Game Over!");
}