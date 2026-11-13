//! `tard web`: play any game in a web browser, against a trained model or
//! another person on the same screen, or watch two models play each other.
//!
//! Serves web/index.html. The server keeps no game state: the page sends the
//! game and the moves so far, the server replays them (the rules live only
//! here) and, when asked for a computer move, runs MCTS with the chosen
//! model. Models are found by scanning --models for *.bpk files with a
//! .net.json sidecar; requests can only name models from that scan.
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server};
use tracing::{info, warn};

use tard::evaluator::TardEvaluator;
use tard::games::{Action, Game, TextGame, WebGame};
use tard::mcts::{mcts, select_action, DirichletHParams};
use tard::model::{load_model, model_meta_path, read_model_meta};
use tard::with_game;

const PAGE: &str = include_str!("../../web/index.html");
// Caps what a request may ask for, so one click can't tie up the GPU for hours.
const MAX_ROLLOUTS: usize = 50_000;
const MAX_BODY_BYTES: u64 = 256 * 1024;
// Models kept loaded at once (e.g. both sides of a computer-vs-computer game).
const MAX_LOADED_MODELS: usize = 4;
// How many of the computer's most-searched moves to report.
const SEARCH_SHOWN: usize = 40;

#[derive(clap::Args, Debug)]
pub struct Args {
    /// Where to look for models: model files, or directories searched
    /// recursively (skipping target/ and hidden directories) for *.bpk
    /// files with a .net.json sidecar
    #[arg(long, num_args = 1.., default_value = ".")]
    models: Vec<PathBuf>,

    /// Default MCTS rollouts per computer move (the page can pick others)
    #[arg(long, default_value_t = 3000)]
    rollouts: usize,

    /// Address to listen on. The default only accepts connections from this
    /// machine.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,
}

//////////////////////////// Models ////////////////////////////

#[derive(Serialize, Debug, Clone, PartialEq)]
struct ModelInfo {
    id: String,    // the model file's path, as scanned
    label: String, // e.g. "step 200000"
    group: String, // e.g. "run_1794483901"
    #[serde(skip)]
    game: Game,
    #[serde(skip)]
    checkpoint: bool, // a run's checkpoint, rather than a model saved on its own
}

// Every model under `roots`, newest run and step first.
fn scan_models(roots: &[PathBuf]) -> Vec<ModelInfo> {
    fn walk(p: &Path, out: &mut Vec<ModelInfo>) {
        if p.is_file() {
            if let Some(m) = model_info(p) {
                out.push(m);
            }
            return;
        }
        let Ok(entries) = std::fs::read_dir(p) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() && (name.starts_with('.') || name == "target") {
                continue;
            }
            walk(&path, out);
        }
    }
    let mut out = Vec::new();
    for root in roots {
        walk(root, &mut out);
    }
    // Saved models (e.g. models/) first, then run checkpoints; newest group
    // and step first within each.
    out.sort_by(|a, b| {
        return (a.checkpoint, &b.group, &b.label_key(), &b.id).cmp(&(
            b.checkpoint,
            &a.group,
            &a.label_key(),
            &a.id,
        ));
    });
    out.dedup_by(|a, b| a.id == b.id);
    return out;
}

// A model file's info, or None if it isn't one (no sidecar, unknown game).
fn model_info(p: &Path) -> Option<ModelInfo> {
    if p.extension()? != "bpk" || !model_meta_path(p).exists() {
        return None;
    }
    let game = Game::from_name(&read_model_meta(p).ok()?.game)?;
    let parent = p.parent().unwrap_or(Path::new(""));
    let parent_name = parent
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    // Checkpoints: <run>/checkpoints/step_<N>/model.bpk
    let (group, label) = match parent_name
        .strip_prefix("step_")
        .and_then(|n| n.parse::<usize>().ok())
    {
        Some(step) => {
            let run = parent
                .parent()
                .and_then(|c| c.parent())
                .and_then(|r| r.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            (run, format!("step {}", step))
        }
        // Any other model file, e.g. models/minipente/minipente-200k.bpk:
        // its name, grouped by its directory.
        None => {
            let dir = parent.display().to_string();
            let dir = dir.strip_prefix("./").unwrap_or(&dir);
            let dir = if dir.is_empty() || dir == "." {
                "models"
            } else {
                dir
            };
            (
                dir.to_string(),
                p.file_stem()?.to_string_lossy().to_string(),
            )
        }
    };
    let checkpoint = label.starts_with("step ");
    return Some(ModelInfo {
        id: p.display().to_string(),
        label,
        group,
        game,
        checkpoint,
    });
}

impl ModelInfo {
    // Sorts "step 9" before "step 10".
    fn label_key(&self) -> (usize, String) {
        let step = self
            .label
            .strip_prefix("step ")
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        return (step, self.label.clone());
    }
}

struct State {
    args: Args,
    device: Device,
    models: Vec<ModelInfo>,
    loaded: Vec<(String, TardEvaluator)>, // oldest first
}

impl State {
    fn rescan(&mut self) {
        self.models = scan_models(&self.args.models);
    }

    // The loaded model `id` (which must be a scanned model for `game`).
    fn evaluator(&mut self, id: &str, game: Game) -> Result<&mut TardEvaluator, String> {
        if !self.models.iter().any(|m| m.id == id) {
            self.rescan();
        }
        let m = self
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or(format!("unknown model '{}'", id))?;
        if m.game != game {
            return Err(format!(
                "{} is a {} model, not {}",
                id,
                m.game.name(),
                game.name()
            ));
        }
        if let Some(i) = self.loaded.iter().position(|(k, _)| k == id) {
            return Ok(&mut self.loaded[i].1);
        }
        let path = PathBuf::from(id);
        let (tard, net) = with_game!(game, G => load_model::<G>(&path, &self.device))
            .map_err(|e| format!("can't load {}: {}", id, e))?;
        info!("Loaded {} ({:?})", id, net);
        if self.loaded.len() >= MAX_LOADED_MODELS {
            self.loaded.remove(0);
        }
        self.loaded
            .push((id.to_string(), TardEvaluator::new(&tard)));
        return Ok(&mut self.loaded.last_mut().unwrap().1);
    }
}

//////////////////////////// API ////////////////////////////

// GET /api/games
#[derive(Serialize, Debug)]
struct Catalog {
    games: Vec<GameInfo>,
    rollouts: usize,
    max_rollouts: usize,
}

#[derive(Serialize, Debug)]
struct GameInfo {
    name: &'static str,
    title: &'static str,
    description: &'static str,
    players: [&'static str; 2],
    models: Vec<ModelInfo>,
}

fn about(game: Game) -> (&'static str, &'static str) {
    return match game {
        Game::MiniPente => (
            "MiniPente",
            "Pente on a 9×9 board. Five in a row or five captured pairs wins.",
        ),
        Game::Pente => (
            "Pente",
            "Official Pente: 19×19, White first, tournament opening rule.",
        ),
        Game::Hive => (
            "Hive",
            "Place and move bugs, no board. Surround the enemy queen to win.",
        ),
        Game::Uttt => (
            "Ultimate Tic-Tac-Toe",
            "Nine small boards. Where you play picks the board your opponent must play in.",
        ),
    };
}

// POST /api/play: the game so far, and whether the computer should move next.
#[derive(Deserialize, Debug)]
struct PlayRequest {
    game: String,
    moves: Vec<Action>,
    #[serde(default)]
    ai: bool,
    model: Option<String>, // required with ai
    rollouts: Option<usize>,
}

// The position after the request's moves (and the computer's, if asked).
#[derive(Serialize, Debug)]
struct View {
    game: &'static str,
    players: [&'static str; 2],
    moves: Vec<Action>,   // all moves, the computer's included
    history: Vec<String>, // each move as text
    last: Option<Value>,  // where the last move played, in the position before it
    position: Value,      // WebGame::position_json
    to_move: &'static str,
    legal: Vec<LegalMove>,
    over: bool,
    result: Option<String>, // "<player> wins" or "Draw" once over
    ai: Option<AiMove>,
}

#[derive(Serialize, Debug)]
struct LegalMove {
    action: Action,
    text: String,
    #[serde(flatten)]
    at: Value, // WebGame::action_json
}

#[derive(Serialize, Debug)]
struct AiMove {
    action: Action,
    text: String,
    player: &'static str,
    model: String,
    eval: f32, // the mover's evaluation before its move: -1 (lost) to +1 (won)
    search: Vec<SearchMove>, // most-searched moves first
    rollouts: usize,
    seconds: f64,
}

#[derive(Serialize, Debug)]
struct SearchMove {
    action: Action,
    text: String,
    share: f32, // of the search
    #[serde(flatten)]
    at: Value,
}

// Replays `moves` from the start, rejecting any that isn't legal. Returns
// the position, each move as text, and where the last one played.
fn replay<G: WebGame>(moves: &[Action]) -> Result<(G, Vec<String>, Option<Value>), String> {
    let mut g = G::default();
    let mut history = Vec::with_capacity(moves.len());
    let mut last = None;
    for (i, &a) in moves.iter().enumerate() {
        if g.is_terminal() {
            return Err(format!("move {} comes after the game ended", i + 1));
        }
        if a >= G::ACTN_SPC_SIZE || !g.legal_actions_mask()[a] {
            return Err(format!("move {} ({}) is not legal", i + 1, a));
        }
        history.push(g.action_text(a));
        last = Some(g.action_json(a));
        g.apply_action(a);
    }
    return Ok((g, history, last));
}

fn view<G: WebGame>(
    g: &G,
    moves: Vec<Action>,
    history: Vec<String>,
    last: Option<Value>,
    ai: Option<AiMove>,
) -> View {
    let legal = g
        .legal_actions_mask()
        .iter()
        .enumerate()
        .filter(|(_, &l)| l)
        .map(|(a, _)| LegalMove {
            action: a,
            text: g.action_text(a),
            at: g.action_json(a),
        })
        .collect();
    let to_move = g.active_player();
    let over = g.is_terminal();
    // The terminal value is from the side to move's view.
    let result = over.then(|| {
        let v = g.get_terminal_value();
        let other = if G::PLAYERS[0] == to_move {
            G::PLAYERS[1]
        } else {
            G::PLAYERS[0]
        };
        if v > 0.0 {
            format!("{} wins", to_move)
        } else if v < 0.0 {
            format!("{} wins", other)
        } else {
            "Draw".to_string()
        }
    });
    return View {
        game: G::NAME,
        players: G::PLAYERS,
        moves,
        history,
        last,
        position: g.position_json(),
        to_move,
        legal,
        over,
        result,
        ai,
    };
}

fn play<G: WebGame>(req: PlayRequest, game: Game, state: &mut State) -> Result<View, String> {
    let (mut g, mut history, mut last) = replay::<G>(&req.moves)?;
    let mut moves = req.moves;
    let mut ai = None;
    if req.ai && !g.is_terminal() {
        let model = req.model.ok_or("choose a model for the computer")?;
        let rollouts = req
            .rollouts
            .unwrap_or(state.args.rollouts)
            .clamp(1, MAX_ROLLOUTS);
        let eval = state.evaluator(&model, game)?;
        let start = Instant::now();
        let dhp = DirichletHParams {
            epsilon: 0.0,
            ..Default::default()
        };
        let (visits, value) = mcts(&g, eval, rollouts, &dhp);
        let action = select_action(&visits, true);
        let mut searched: Vec<Action> = (0..visits.len()).filter(|&a| visits[a] >= 0.005).collect();
        searched.sort_by(|&a, &b| visits[b].total_cmp(&visits[a]));
        searched.truncate(SEARCH_SHOWN);
        let search = searched
            .into_iter()
            .map(|a| SearchMove {
                action: a,
                text: g.action_text(a),
                share: visits[a],
                at: g.action_json(a),
            })
            .collect();
        let text = g.action_text(action);
        let player = g.active_player();
        last = Some(g.action_json(action));
        g.apply_action(action);
        moves.push(action);
        history.push(text.clone());
        let seconds = start.elapsed().as_secs_f64();
        info!(
            "{} {} plays {} after {} rollouts in {:.1}s, eval {:+.2}",
            game.name(),
            player,
            text,
            rollouts,
            seconds,
            value
        );
        ai = Some(AiMove {
            action,
            text,
            player,
            model,
            eval: value,
            search,
            rollouts,
            seconds,
        });
    }
    return Ok(view(&g, moves, history, last, ai));
}

fn catalog(state: &mut State) -> Catalog {
    state.rescan();
    let games = Game::ALL
        .iter()
        .map(|&game| {
            let (title, description) = about(game);
            GameInfo {
                name: game.name(),
                title,
                description,
                players: with_game!(game, G => G::PLAYERS),
                models: state
                    .models
                    .iter()
                    .filter(|m| m.game == game)
                    .cloned()
                    .collect(),
            }
        })
        .collect();
    return Catalog {
        games,
        rollouts: state.args.rollouts.clamp(1, MAX_ROLLOUTS),
        max_rollouts: MAX_ROLLOUTS,
    };
}

//////////////////////////// HTTP ////////////////////////////

type Reply = Response<std::io::Cursor<Vec<u8>>>;

fn json_response<T: Serialize>(status: u16, body: &T) -> Reply {
    let text = serde_json::to_string(body).expect("serializable response");
    return Response::from_string(text)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
}

fn error_response(status: u16, message: &str) -> Reply {
    return json_response(status, &serde_json::json!({ "error": message }));
}

fn handle(req: &mut Request, state: &mut State) -> Reply {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    return match (req.method(), path.as_str()) {
        (Method::Get, "/") => Response::from_string(PAGE)
            .with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap()),
        (Method::Get, "/api/games") => json_response(200, &catalog(state)),
        (Method::Post, "/api/play") => {
            let mut body = String::new();
            if let Err(e) = req
                .as_reader()
                .take(MAX_BODY_BYTES)
                .read_to_string(&mut body)
            {
                return error_response(400, &format!("can't read request: {}", e));
            }
            let parsed: PlayRequest = match serde_json::from_str(&body) {
                Ok(p) => p,
                Err(e) => return error_response(400, &format!("bad request: {}", e)),
            };
            let Some(game) = Game::from_name(&parsed.game) else {
                return error_response(400, &format!("unknown game '{}'", parsed.game));
            };
            match with_game!(game, G => play::<G>(parsed, game, state)) {
                Ok(v) => json_response(200, &v),
                Err(e) => error_response(400, &e),
            }
        }
        _ => error_response(404, "not found"),
    };
}

pub fn run(args: Args) {
    let addr = format!("{}:{}", args.host, args.port);
    let mut state = State {
        args,
        device: Device::default(),
        models: Vec::new(),
        loaded: Vec::new(),
    };
    state.rescan();
    for game in Game::ALL {
        let n = state.models.iter().filter(|m| m.game == game).count();
        info!("{} models for {}", n, game.name());
    }
    let server = Server::http(&addr).unwrap_or_else(|e| panic!("Can't listen on {}: {}", addr, e));
    println!("Play at http://{}/  (Ctrl-C to stop)", addr);
    // One request at a time: one GPU, and usually one player.
    for mut req in server.incoming_requests() {
        let response = handle(&mut req, &mut state);
        if let Err(e) = req.respond(response) {
            warn!("Failed to send a response: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tard::games::hive::HiveState;
    use tard::games::minipente::{coord_2_action, MiniPenteState};
    use tard::games::pente::PenteState;

    fn scratch_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tard_web_test_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        return d;
    }

    // Black's five on row 0 in nine moves.
    fn black_wins() -> Vec<Action> {
        return [
            (0, 0),
            (8, 0),
            (0, 1),
            (8, 2),
            (0, 2),
            (8, 4),
            (0, 3),
            (8, 6),
            (0, 4),
        ]
        .iter()
        .map(|&(r, c)| coord_2_action(r, c))
        .collect();
    }

    #[test]
    fn replay_rejects_illegal_and_late_moves() {
        let a = coord_2_action(4, 4);
        assert!(replay::<MiniPenteState>(&[a, coord_2_action(4, 5)]).is_ok());
        assert!(replay::<MiniPenteState>(&[a, a])
            .err()
            .unwrap()
            .contains("move 2"));
        assert!(replay::<MiniPenteState>(&[81]).is_err(), "off the board");
        let mut late = black_wins();
        late.push(coord_2_action(5, 5));
        assert!(replay::<MiniPenteState>(&late)
            .err()
            .unwrap()
            .contains("after the game ended"));
        // Pente's first move must be the center.
        assert!(replay::<PenteState>(&[0]).is_err());
    }

    #[test]
    fn grid_view_shows_board_history_and_result() {
        let moves = vec![coord_2_action(4, 4), coord_2_action(0, 8)];
        let (g, history, last) = replay::<MiniPenteState>(&moves).unwrap();
        let v = view(&g, moves, history, last, None);
        assert_eq!(v.position["kind"], "grid");
        assert_eq!(v.position["board"][4], "....B....");
        assert_eq!(v.position["board"][0], "........W");
        assert_eq!(v.history, ["4, 4", "0, 8"]);
        assert_eq!(v.last, Some(serde_json::json!({ "row": 0, "col": 8 })));
        assert_eq!(v.to_move, "Black");
        assert_eq!(v.legal.len(), 79);
        assert!(!v.over && v.result.is_none());

        let (g, history, last) = replay::<MiniPenteState>(&black_wins()).unwrap();
        let v = view(&g, black_wins(), history, last, None);
        assert_eq!(v.result.as_deref(), Some("Black wins"));
        assert!(v.legal.is_empty());
    }

    #[test]
    fn hive_view_lists_hands_and_moves_by_piece() {
        let (g, history, last) = replay::<HiveState>(&[]).unwrap();
        let v = view(&g, vec![], history, last, None);
        assert_eq!(v.position["kind"], "hive");
        assert_eq!(v.position["hands"]["White"].as_array().unwrap().len(), 11);
        // First move: any piece but the queen, all onto the one center cell.
        let pieces: Vec<&str> = v
            .legal
            .iter()
            .map(|l| l.at["piece"].as_str().unwrap())
            .collect();
        assert_eq!(pieces, ["wS1", "wB1", "wG1", "wA1"]);
        assert!(v
            .legal
            .iter()
            .all(|l| l.at["q"] == v.legal[0].at["q"] && l.at["r"] == v.legal[0].at["r"]));

        let first = v.legal[3].action;
        let (g, history, last) = replay::<HiveState>(&[first]).unwrap();
        let v = view(&g, vec![first], history, last, None);
        assert_eq!(v.history, ["wA1"]);
        assert_eq!(v.position["cells"][0]["stack"], serde_json::json!(["wA1"]));
        assert_eq!(v.to_move, "Black");
    }

    #[test]
    fn uttt_view_shows_the_forced_board() {
        use tard::games::uttt::{coord_2_action as at, UtttState};
        let moves = vec![at(1, 1)];
        let (g, history, last) = replay::<UtttState>(&moves).unwrap();
        let v = view(&g, moves, history, last, None);
        assert_eq!(v.position["kind"], "uttt");
        assert_eq!(v.position["board"][1], ".X.......");
        assert_eq!(
            v.position["next"], 4,
            "the middle cell sends O to the center board"
        );
        assert_eq!(v.position["boards"][0], ".");
        assert_eq!(v.legal.len(), 9);
        assert!(v
            .legal
            .iter()
            .all(|l| (3..6).contains(&l.at["row"].as_u64().unwrap())));
        assert_eq!(v.to_move, "O");
    }

    #[test]
    fn scan_finds_models_with_sidecars_newest_first() {
        let root = scratch_dir("scan");
        let meta = |game: &str| {
            format!(
                r#"{{"game": "{}", "channels": 8, "blocks": 1, "value_hidden": 8}}"#,
                game
            )
        };
        for (run, step, game) in [
            ("run_100", 512, "minipente"),
            ("run_100", 4096, "minipente"),
            ("run_200", 512, "hive"),
        ] {
            let d = root
                .join(run)
                .join("checkpoints")
                .join(format!("step_{:06}", step));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("model.bpk"), "weights").unwrap();
            std::fs::write(d.join("model.net.json"), meta(game)).unwrap();
            std::fs::write(d.join("optim.bpk"), "no sidecar").unwrap();
        }
        std::fs::write(root.join("loose.bpk"), "").unwrap();
        std::fs::write(root.join("loose.net.json"), meta("pente")).unwrap();
        std::fs::write(root.join("chess.bpk"), "").unwrap();
        std::fs::write(root.join("chess.net.json"), meta("chess")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("target").join("x.bpk"), "").unwrap();
        std::fs::write(root.join("target").join("x.net.json"), meta("pente")).unwrap();

        let models = scan_models(&[root.clone()]);
        let summary: Vec<(String, String, Game)> = models
            .iter()
            .map(|m| (m.group.clone(), m.label.clone(), m.game))
            .collect();
        let dir = root.display().to_string();
        assert_eq!(
            summary,
            [
                (dir, "loose".to_string(), Game::Pente),
                ("run_200".to_string(), "step 512".to_string(), Game::Hive),
                (
                    "run_100".to_string(),
                    "step 4096".to_string(),
                    Game::MiniPente
                ),
                (
                    "run_100".to_string(),
                    "step 512".to_string(),
                    Game::MiniPente
                ),
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
