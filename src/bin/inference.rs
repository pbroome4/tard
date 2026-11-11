
use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::{Device};
use tracing::{info};
use tracing_subscriber::filter::{LevelFilter, EnvFilter};
use std::io::{self, Write};
use std::ptr::read;

use pentard::evaluator::TardEvaluator;
use pentard::gamestate::GameState;
use pentard::pente::{PenteState, action_2_coord};
use pentard::tard::Tard;
use pentard::utils::*;
use pentard::engine::*;

fn main(){
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
    let mut tard = Tard::new(PenteState::N_CHAN,
                             PenteState::ROWS,
                             PenteState::COLS,
                             PenteState::ACTN_SPC_SIZE,
                             &device);
    let model_file = "model.bpk";
    info!("Loading model record: {}", model_file);
    let record = ModuleRecord::load(model_file).unwrap();
    info!("Loading record into tard");
    tard = tard.load_record(record);
    info!("Tard model loaded");
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