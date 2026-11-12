use std::path::{Path, PathBuf};
use tracing::{trace, debug, info};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use burn::tensor::{Tensor, Device,TensorData};
use burn::store::{ModuleRecord, RecordError};
use burn::module::Module;

use crate::gamestate::GameState;
use crate::tard::{Tard, TrainingElement};
use crate::engine::SelfPlayElement;
use crate::pente::PenteState;

#[derive(Debug)]
pub struct OutputManager{
    pub basepath_d: PathBuf,
    pub model_d: PathBuf,
    pub loss_f: PathBuf
}

impl OutputManager{
    /// Creates a fresh run directory `<root>/run_<unix_seconds>/` containing
    /// `models/` and `loss.csv`. Never deletes or overwrites earlier runs.
    pub fn new<P: Into<PathBuf>>(root: P) -> io::Result<Self>{
        let root_d = root.into();
        fs::create_dir_all(&root_d)?;
        let secs = SystemTime::now().duration_since(UNIX_EPOCH)
                                    .map(|d| d.as_secs())
                                    .unwrap_or(0);
        // create_dir (not create_dir_all) fails if the directory already
        // exists, so we never write into another run's directory.
        let mut basepath_d = root_d.join(format!("run_{}", secs));
        let mut suffix = 1;
        while let Err(e) = fs::create_dir(&basepath_d){
            if e.kind() != io::ErrorKind::AlreadyExists{
                return Err(e);
            }
            basepath_d = root_d.join(format!("run_{}_{}", secs, suffix));
            suffix += 1;
        }
        info!("Created run directory {}", basepath_d.display());
        let model_d = basepath_d.join("models");
        let loss_f = basepath_d.join("loss.csv");
        fs::create_dir(&model_d)?;
        let mut lf = OpenOptions::new().create_new(true)
                                  .write(true)
                                  .open(&loss_f)?;
        lf.write_all(b"loss, cross_entropy_loss, mse_loss, learning_rate\n")?;

        let s = Self {
            basepath_d: basepath_d,
            model_d: model_d,
            loss_f: loss_f
        };
        info!("Created OutputManager: {:?}", s);
        return Ok(s);
    }
}


pub fn selfplay_to_training<G:GameState>(sp_vec: Vec<SelfPlayElement<G>>) -> Vec<TrainingElement>{
    let mut te_vec = Vec::with_capacity(sp_vec.len());
    for spe in sp_vec.into_iter(){
        let state_td = TensorData::new(spe.g.to_tensor(),
                                   [G::N_CHAN, G::ROWS, G::COLS]);
        let mcts_td = TensorData::new(spe.mcts_post, [G::ACTN_SPC_SIZE]);
        let outcome_td = TensorData::new(vec![spe.outcome], [1]);
        let te = TrainingElement{
            state: state_td,
            mcts_posterior: mcts_td,
            outcome: outcome_td,
        };
        te_vec.push(te);
    }
    return te_vec;
}



pub fn parse_coord(s: &str) -> Result<(usize, usize), String> {
    let mut parts = s.trim().split(',');
    if let (Some(row_str), Some(col_str)) = (parts.next(), parts.next()) {
        let x: Result<usize, _> = row_str.trim().parse();
        let y: Result<usize, _> = col_str.trim().parse();
        match (x, y) {
            (Ok(row), Ok(col)) => {
                return Ok((row, col));
            }
            _ =>{
                return Err(String::from("Invalid format"));
            } 
        }
    }
    return Err(String::from("Invalid format"));
}


pub fn read_move() -> Result<(usize, usize), String>{
    let mut input_buffer = String::new();
    io::stdin().read_line(&mut input_buffer).unwrap();
    match parse_coord(&input_buffer){
        Ok((r,c)) => {
            return Ok((r,c));
        }
        Err(s) =>{
            return Err(s);
        } 
    }
}

pub fn read_yn() -> bool{
    let mut input_buffer = String::new();
    io::stdin().read_line(&mut input_buffer).unwrap();
    if input_buffer.contains("y"){
        return true;
    }
    return false;
    
} 


pub fn legalize_policy(mut p: Vec<f32>, g: &impl GameState) -> Vec<f32>{
    let legal_mask = g.legal_actions_mask();
    for i in 0..p.len(){
        if legal_mask[i] == 0.0{
            p[i] = 0.0;
        }
    }
    return normalize(p);
}


// normalizes probability vector to 1.0.
// Assumes sum is non-zero and that each element >=0
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let sum:f32 = v.iter().sum();
    for e in v.iter_mut(){
        *e = *e/sum;
    }
    return v;
}


pub fn load_model(p: &Path, device: &Device) -> Result<Tard, RecordError>{
    let mut tard = Tard::new(PenteState::N_CHAN,
                             PenteState::ROWS,
                             PenteState::COLS,
                             PenteState::ACTN_SPC_SIZE,
                             &device);
    info!("Loading model record: {}", p.display());
    let record = ModuleRecord::load(p)?;
    info!("Loading record into tard");
    tard = tard.load_record(record);
    info!("Tard model loaded");
    return Ok(tard);
}