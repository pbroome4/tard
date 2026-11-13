//! A training run's output directory: config.json, metrics.csv and
//! checkpoints.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

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
    // Throughput over the iteration (from its start to just before this row)
    pub iter_time_s: f64,     // wall time of the whole iteration
    pub selfplay_wait_s: f64, // main thread waiting for self-play games
    pub train_s: f64,         // main thread running training steps
    pub evals: u64,           // network evaluations done by self-play
    pub eval_batches: u64,    // forward passes those evaluations took
    pub eval_busy_s: f64,     // time inside those forward passes (summed over threads)
}

const METRICS_HEADER: &str = concat!(
    "step,wall_time_s,lr,loss,policy_loss,value_loss,games,positions,mean_game_len,draw_rate,",
    "iter_time_s,selfplay_wait_s,train_s,positions_per_s,evals_per_s,mean_eval_batch,eval_busy_frac\n");

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
        let secs = m.iter_time_s.max(1e-9);
        writeln!(self.metrics_f, "{},{:.1},{},{},{},{},{},{},{:.2},{:.4},{:.1},{:.1},{:.1},{:.1},{:.0},{:.1},{:.3}",
                 m.step, m.wall_time_s, m.lr, m.loss, m.policy_loss, m.value_loss,
                 m.games, m.positions, m.positions as f64 / games, m.draws as f64 / games,
                 m.iter_time_s, m.selfplay_wait_s, m.train_s,
                 m.positions as f64 / secs,
                 m.evals as f64 / secs,
                 m.evals as f64 / m.eval_batches.max(1) as f64,
                 m.eval_busy_s / secs)?;
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


#[cfg(test)]
mod tests{
    use super::*;

    // A fresh, empty scratch directory under the system temp dir.
    fn scratch_dir(name: &str) -> PathBuf{
        let d = std::env::temp_dir().join(format!("tard_test_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        return d;
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
            iter_time_s: 10.0, selfplay_wait_s: 7.5, train_s: 2.0,
            evals: 50_000, eval_batches: 250, eval_busy_s: 6.0,
        }).unwrap();
        let csv = fs::read_to_string(om.run_d.join("metrics.csv")).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        let n_cols = lines[0].split(',').count();
        assert_eq!(lines[1].split(',').count(), n_cols);
        assert_eq!(lines[1], "512,12.3,0.005,1.5,1,0.5,4,130,32.50,0.2500,10.0,7.5,2.0,13.0,5000,200.0,0.600");
        fs::remove_dir_all(&root).unwrap();
    }
}
