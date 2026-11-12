// Play MiniPente or Pente against a trained model. The game comes from the
// model's .net.json sidecar.
use burn::tensor::{Device};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use std::io::{self, Write};

use tard::evaluator::TardEvaluator;
use tard::gamestate::{GameState, GridGame};
use tard::minipente::MiniPenteState;
use tard::pente::PenteState;
use tard::utils::*;
use tard::engine::*;

#[derive(Parser, Debug)]
#[command(about = "Play MiniPente or Pente against a trained Tard model")]
struct Args{
    /// Model file to play against (its <name>.net.json sidecar, which names
    /// the game, must sit next to it)
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
    filter = filter.add_directive("tard=info".parse().unwrap());
    filter = filter.add_directive("inference=info".parse().unwrap());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    let meta = read_model_meta(&args.model).unwrap_or_else(|e|{
        panic!("Failed to read model {}: {}", args.model.display(), e)
    });
    match meta.game.as_str(){
        MiniPenteState::NAME => play::<MiniPenteState>(&args),
        PenteState::NAME => play::<PenteState>(&args),
        other => panic!("Model {} is for game '{}', which this bin can't play", args.model.display(), other),
    }
}

fn play<G: GridGame>(args: &Args){
    info!("Starting inference ({})", G::NAME);
    let device = Device::default();
    info!("Device: {:?}", &device);
    let (tard, net) = load_model::<G>(&args.model, &device).unwrap_or_else(|e|{
        panic!("Failed to load model {}: {}", args.model.display(), e)
    });
    info!("Network: {:?}", net);
    let mut eval = TardEvaluator::new(&tard);

    let mut g = G::default();
    print!("Is human first to play? (y/n) ");
    io::stdout().flush().unwrap();
    let Some(mut is_player_move) = read_yn() else{
        println!("\nInput closed, quitting");
        return;
    };
    println!("\n{}", g);
    loop{
        if is_player_move{
            let player: &str = g.active_player();
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
            if !g.play_at(row, col){
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
            let player: &str = g.active_player();
            println!("Computer thinking of move for {player}");
            let (action, val) = inference(&g, &mut eval, args.rollouts);
            let (row, col) = G::action_coord(action);
            assert!(g.play_at(row, col), "Engine chose an illegal move ({}, {})", row, col);
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
