use std::path::{Path, PathBuf};
use tracing::{info, warn};
use std::any::Any;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use burn::tensor::Device;
use burn::store::{ModuleRecord, RecordError};
use burn::module::Module;

use crate::gamestate::GameState;
use crate::tard::{Tard, TrainingElement};
use crate::engine::SelfPlayElement;

/// Layout of one training run:
///
/// ```text
/// <root>/run_<unix_seconds>/
///   config.json            the run's settings (CLI args), written at start
///   metrics.csv            one row per training iteration, keyed by step
///   checkpoints/
///     step_000512/
///       model.bpk          network weights
///       optim.bpk          optimizer state (needed to --resume)
/// ```
#[derive(Debug)]
pub struct OutputManager{
    pub run_d: PathBuf,
    checkpoints_d: PathBuf,
    metrics_f: File,
}

/// One metrics.csv row: training and self-play stats for one iteration.
pub struct IterationMetrics{
    pub step: usize,         // training step at the end of the iteration
    pub wall_time_s: f64,    // seconds since the run started
    pub lr: f64,             // learning rate of the iteration's last step
    pub loss: f32,           // mean over the iteration's steps
    pub policy_loss: f32,    // mean cross-entropy against the MCTS posterior
    pub value_loss: f32,     // mean squared error against the game outcome
    pub games: usize,        // self-play games consumed this iteration
    pub positions: usize,    // positions from those games
    pub draws: usize,        // how many of those games were draws
}

const METRICS_HEADER: &str =
    "step,wall_time_s,lr,loss,policy_loss,value_loss,games,positions,mean_game_len,draw_rate\n";

impl OutputManager{
    /// Creates a fresh run directory under `root` (see the layout above) and
    /// writes `config_json` into it. Never deletes or overwrites earlier runs.
    pub fn new<P: Into<PathBuf>>(root: P, config_json: &str) -> io::Result<Self>{
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
        fs::write(basepath_d.join("config.json"), config_json)?;
        let checkpoints_d = basepath_d.join("checkpoints");
        fs::create_dir(&checkpoints_d)?;
        let mut metrics_f = OpenOptions::new().create_new(true)
                                              .write(true)
                                              .open(basepath_d.join("metrics.csv"))?;
        metrics_f.write_all(METRICS_HEADER.as_bytes())?;
        return Ok(Self {
            run_d: basepath_d,
            checkpoints_d,
            metrics_f,
        });
    }

    /// Creates and returns `checkpoints/step_<step>/` (zero-padded so
    /// directory listings sort by step).
    pub fn checkpoint_dir(&self, step: usize) -> io::Result<PathBuf>{
        let d = self.checkpoints_d.join(format!("step_{:06}", step));
        fs::create_dir(&d)?;
        return Ok(d);
    }

    /// Appends one row to metrics.csv.
    pub fn append_metrics(&mut self, m: &IterationMetrics) -> io::Result<()>{
        let games = m.games.max(1) as f64;
        writeln!(self.metrics_f, "{},{:.1},{},{},{},{},{},{},{:.2},{:.4}",
                 m.step, m.wall_time_s, m.lr, m.loss, m.policy_loss, m.value_loss,
                 m.games, m.positions, m.positions as f64 / games, m.draws as f64 / games)?;
        return self.metrics_f.flush();
    }
}

/// Parses <N> from a checkpoint directory named `step_<N>`.
pub fn step_from_checkpoint_dir(p: &Path) -> Option<usize>{
    return p.file_name()?
            .to_str()?
            .strip_prefix("step_")?
            .parse()
            .ok();
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

#[cfg(test)]
mod tests{
    use super::*;
    use crate::pente::PenteState;

    // A fresh, empty scratch directory under the system temp dir.
    fn scratch_dir(name: &str) -> PathBuf{
        let d = std::env::temp_dir().join(format!("pentard_test_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        return d;
    }

    #[test]
    fn normalize_scales_to_sum_one(){
        assert_eq!(normalize(vec![1.0, 3.0]), vec![0.25, 0.75]);
    }

    #[test]
    fn legalize_policy_masks_illegal_moves_and_renormalizes(){
        let mut g = PenteState::new();
        g.play(0, 0);
        let mut p = vec![0.0; 81];
        p[0] = 0.5; // occupied
        p[1] = 0.25;
        p[2] = 0.25;
        let p = legalize_policy(p, &g);
        assert_eq!(p[0], 0.0);
        assert!((p[1] - 0.5).abs() < 1e-6 && (p[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn legalize_policy_falls_back_to_uniform_without_usable_mass(){
        let mut g = PenteState::new();
        g.play(4, 4);
        let all_on_illegal = {
            let mut p = vec![0.0; 81];
            p[40] = 1.0;
            p
        };
        for bad in [vec![0.0; 81], vec![f32::NAN; 81], vec![f32::INFINITY; 81], all_on_illegal]{
            let p = legalize_policy(bad, &g);
            assert_eq!(p[40], 0.0);
            for (a, &x) in p.iter().enumerate(){
                if a != 40{
                    assert!((x - 1.0 / 80.0).abs() < 1e-6, "action {a}: {x}");
                }
            }
        }
    }

    #[test]
    fn parse_coord_accepts_row_comma_col(){
        assert_eq!(parse_coord("3,4"), Ok((3, 4)));
        assert_eq!(parse_coord(" 3 , 4 \n"), Ok((3, 4)));
        assert!(parse_coord("3").is_err());
        assert!(parse_coord("a,b").is_err());
        assert!(parse_coord("-1,2").is_err());
        assert!(parse_coord("").is_err());
    }

    #[test]
    fn step_from_checkpoint_dir_parses_step_dirs_only(){
        assert_eq!(step_from_checkpoint_dir(Path::new("out/run_1/checkpoints/step_000512")), Some(512));
        assert_eq!(step_from_checkpoint_dir(Path::new("step_7")), Some(7));
        assert_eq!(step_from_checkpoint_dir(Path::new("step_000512/")), Some(512));
        assert_eq!(step_from_checkpoint_dir(Path::new("model_512")), None);
        assert_eq!(step_from_checkpoint_dir(Path::new("step_abc")), None);
    }

    #[test]
    fn selfplay_to_training_keeps_targets_and_encodes_state(){
        let g = PenteState::new();
        let spe = SelfPlayElement{ g: g.clone(), mcts_post: vec![0.5, 0.5], outcome: -1.0 };
        let te = selfplay_to_training(vec![spe]);
        assert_eq!(te.len(), 1);
        assert_eq!(te[0].state, g.to_tensor());
        assert_eq!(te[0].mcts_posterior, vec![0.5, 0.5]);
        assert_eq!(te[0].outcome, -1.0);
    }

    #[test]
    fn panic_message_reads_str_and_string_payloads(){
        let p = std::panic::catch_unwind(|| panic!("static msg")).unwrap_err();
        assert_eq!(panic_message(&p), "static msg");
        let p = std::panic::catch_unwind(|| panic!("formatted {}", 7)).unwrap_err();
        assert_eq!(panic_message(&p), "formatted 7");
    }

    #[test]
    fn output_manager_creates_the_run_layout(){
        let root = scratch_dir("layout");
        let om = OutputManager::new(&root, "{\"workers\": 4}").unwrap();
        assert_eq!(om.run_d.parent().unwrap(), root);
        assert!(om.run_d.file_name().unwrap().to_str().unwrap().starts_with("run_"));
        assert_eq!(fs::read_to_string(om.run_d.join("config.json")).unwrap(), "{\"workers\": 4}");
        assert_eq!(fs::read_to_string(om.run_d.join("metrics.csv")).unwrap(), METRICS_HEADER);

        let ckpt = om.checkpoint_dir(512).unwrap();
        assert_eq!(ckpt, om.run_d.join("checkpoints").join("step_000512"));
        assert!(ckpt.is_dir());
        assert_eq!(step_from_checkpoint_dir(&ckpt), Some(512));
        assert!(om.checkpoint_dir(512).is_err(), "an existing checkpoint must not be reused");

        // A second run in the same second still gets its own directory.
        let om2 = OutputManager::new(&root, "{}").unwrap();
        assert_ne!(om.run_d, om2.run_d);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn append_metrics_writes_one_csv_row_per_call(){
        let root = scratch_dir("metrics");
        let mut om = OutputManager::new(&root, "{}").unwrap();
        om.append_metrics(&IterationMetrics{
            step: 512, wall_time_s: 12.34, lr: 0.005,
            loss: 1.5, policy_loss: 1.0, value_loss: 0.5,
            games: 4, positions: 130, draws: 1,
        }).unwrap();
        let csv = fs::read_to_string(om.run_d.join("metrics.csv")).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        let n_cols = lines[0].split(',').count();
        assert_eq!(lines[1].split(',').count(), n_cols);
        assert_eq!(lines[1], "512,12.3,0.005,1.5,1,0.5,4,130,32.50,0.2500");
        fs::remove_dir_all(&root).unwrap();
    }
}
