use std::path::{Path, PathBuf};
use tracing::{info, warn};
use std::any::Any;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use burn::tensor::Device;
use burn::store::{ModuleRecord, RecordError};
use burn::module::Module;

use crate::gamestate::GameState;
use crate::tard::{Tard, TrainingElement};
use crate::engine::SelfPlayElement;

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
            basepath_d,
            model_d,
            loss_f
        };
        info!("Created OutputManager: {:?}", s);
        return Ok(s);
    }
}


pub fn selfplay_to_training<G:GameState>(sp_vec: Vec<SelfPlayElement<G>>) -> Vec<TrainingElement>{
    let mut te_vec = Vec::with_capacity(sp_vec.len());
    for spe in sp_vec.into_iter(){
        let te = TrainingElement{
            state: spe.g.to_tensor(),
            mcts_posterior: spe.mcts_post,
            outcome: spe.outcome,
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


#[derive(Debug)]
pub enum ReadError{
    Eof,             // stdin closed (or unreadable); callers should stop prompting
    Invalid(String), // line read but couldn't be parsed
}

// Reads one line from stdin. None on EOF or read error.
fn read_line() -> Option<String>{
    let mut input_buffer = String::new();
    match io::stdin().read_line(&mut input_buffer){
        Ok(0) | Err(_) => None,
        Ok(_) => Some(input_buffer),
    }
}

pub fn read_move() -> Result<(usize, usize), ReadError>{
    let line = read_line().ok_or(ReadError::Eof)?;
    return parse_coord(&line).map_err(ReadError::Invalid);
}

// Reads a y/n answer, re-prompting on anything else. None on EOF.
pub fn read_yn() -> Option<bool>{
    loop{
        let line = read_line()?;
        match line.trim().to_lowercase().as_str(){
            "y" | "yes" => return Some(true),
            "n" | "no" => return Some(false),
            _ => {
                print!("Please answer y or n: ");
                let _ = io::stdout().flush();
            }
        }
    }
}

// Best-effort text of a panic payload (from JoinHandle::join or catch_unwind).
pub fn panic_message(payload: &Box<dyn Any + Send>) -> &str{
    if let Some(s) = payload.downcast_ref::<&str>(){
        return s;
    }
    if let Some(s) = payload.downcast_ref::<String>(){
        return s;
    }
    return "<non-string panic payload>";
}


pub fn legalize_policy(mut p: Vec<f32>, g: &impl GameState) -> Vec<f32>{
    let legal_mask = g.legal_actions_mask();
    for i in 0..p.len(){
        if !legal_mask[i]{
            p[i] = 0.0;
        }
    }
    let sum: f32 = p.iter().sum();
    if !(sum > 0.0 && sum.is_finite()){
        // All legal moves got zero (underflow) or non-finite (NaN/inf)
        // probability. Fall back to uniform over legal moves rather than
        // producing NaNs.
        warn!("Policy has no usable mass over legal moves (sum = {}); using uniform", sum);
        p = legal_mask.iter().map(|&legal| if legal {1.0} else {0.0}).collect();
    }
    return normalize(p);
}


// normalizes probability vector to 1.0.
// Assumes sum is non-zero and that each element >=0
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let sum:f32 = v.iter().sum();
    for e in v.iter_mut(){
        *e /= sum;
    }
    return v;
}


// Loads a checkpoint into a network sized for game G.
pub fn load_model<G: GameState>(p: &Path, device: &Device) -> Result<Tard, RecordError>{
    let mut tard = Tard::for_game::<G>(device);
    info!("Loading model record: {}", p.display());
    let record = ModuleRecord::load(p)?;
    info!("Loading record into tard");
    tard = tard.load_record(record);
    info!("Tard model loaded");
    return Ok(tard);
}