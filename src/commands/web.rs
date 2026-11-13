//! `tard web`: play MiniPente against a trained model in a web browser.
//! Serves web/minipente.html on localhost; the page sends the moves so far,
//! and the server replays them (the rules live only here) and runs the
//! model's MCTS for the computer's moves.
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server};
use tracing::{info, warn};

use tard::evaluator::TardEvaluator;
use tard::games::minipente::{action_2_coord, MiniPenteState, Piece, COLS, ROWS};
use tard::games::{Action, GameState};
use tard::mcts::{mcts, select_action, DirichletHParams};
use tard::model::load_model;

const PAGE: &str = include_str!("../../web/minipente.html");
// Caps what a request may ask for, so one click can't tie up the GPU for hours.
const MAX_ROLLOUTS: usize = 50_000;
const MAX_BODY_BYTES: u64 = 64 * 1024;

#[derive(clap::Args, Debug)]
pub struct Args{
    /// MiniPente model file (its <name>.net.json sidecar must sit next to it)
    #[arg(long, default_value = "model.bpk")]
    model: PathBuf,

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

// POST /api/play: the game so far, and whether the computer should move next.
#[derive(Deserialize, Debug)]
struct PlayRequest{
    moves: Vec<Action>,
    #[serde(default)]
    ai: bool,
    rollouts: Option<usize>,
}

// The position after the request's moves (and the computer's, if asked).
#[derive(Serialize, Debug)]
struct View{
    moves: Vec<Action>,       // all moves, the computer's included
    board: Vec<String>,       // one string per row: '.', 'B' or 'W' per point
    black_captures: u8,       // pairs captured by Black
    white_captures: u8,
    to_move: &'static str,    // "Black" or "White"
    legal: Vec<bool>,         // per action (row * COLS + col)
    over: bool,
    result: Option<String>,   // "Black wins", "White wins" or "Draw" once over
    ai: Option<AiMove>,
}

#[derive(Serialize, Debug)]
struct AiMove{
    row: usize,
    col: usize,
    eval: f32,         // the computer's evaluation before its move, from its side: -1 (lost) to +1 (won)
    visits: Vec<f32>,  // share of the search spent on each action
    rollouts: usize,
    seconds: f64,
}

#[derive(Serialize, Debug)]
struct ServerInfo{
    model: String,
    network: String,
    rollouts: usize,
    max_rollouts: usize,
    rows: usize,
    cols: usize,
}

// Replays `moves` from the start, rejecting any that isn't legal.
fn replay(moves: &[Action]) -> Result<MiniPenteState, String>{
    let mut g = MiniPenteState::new();
    for (i, &a) in moves.iter().enumerate(){
        if g.is_terminal(){
            return Err(format!("move {} comes after the game ended", i + 1));
        }
        if a >= MiniPenteState::ACTN_SPC_SIZE || !g.is_legal_action(a){
            return Err(format!("move {} ({}) is not legal", i + 1, a));
        }
        g.apply_action(a);
    }
    return Ok(g);
}

fn view(g: &MiniPenteState, moves: Vec<Action>, ai: Option<AiMove>) -> View{
    let board = (0..ROWS).map(|r| (0..COLS).map(|c| match g.piece(r, c){
        Piece::Empty => '.',
        Piece::Black => 'B',
        Piece::White => 'W',
    }).collect()).collect();
    let (black_captures, white_captures) = g.captures();
    let over = g.is_terminal();
    // The terminal value is from the side to move's view, and in Pente only
    // the player who just moved can have won.
    let result = over.then(|| match (g.get_terminal_value() == 0.0, g.is_black()){
        (true, _) => "Draw",
        (false, true) => "White wins",
        (false, false) => "Black wins",
    }.to_string());
    return View{
        moves,
        board,
        black_captures,
        white_captures,
        to_move: if g.is_black() {"Black"} else {"White"},
        legal: g.legal_actions_mask(),
        over,
        result,
        ai,
    };
}

fn play(req: PlayRequest, eval: &mut TardEvaluator, default_rollouts: usize) -> Result<View, String>{
    let mut g = replay(&req.moves)?;
    let mut moves = req.moves;
    let mut ai = None;
    if req.ai && !g.is_terminal(){
        let rollouts = req.rollouts.unwrap_or(default_rollouts).clamp(1, MAX_ROLLOUTS);
        let start = Instant::now();
        let dhp = DirichletHParams{ epsilon: 0.0, ..Default::default() };
        let (visits, value) = mcts(&g, eval, rollouts, &dhp);
        let action = select_action(&visits, true);
        g.apply_action(action);
        moves.push(action);
        let (row, col) = action_2_coord(action);
        let seconds = start.elapsed().as_secs_f64();
        info!("Computer plays ({}, {}) after {} rollouts in {:.1}s, eval {:+.2}", row, col, rollouts, seconds, value);
        ai = Some(AiMove{ row, col, eval: value, visits, rollouts, seconds });
    }
    return Ok(view(&g, moves, ai));
}

fn json_response<T: Serialize>(status: u16, body: &T) -> Response<std::io::Cursor<Vec<u8>>>{
    let text = serde_json::to_string(body).expect("serializable response");
    return Response::from_string(text)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
}

fn error_response(status: u16, message: &str) -> Response<std::io::Cursor<Vec<u8>>>{
    return json_response(status, &serde_json::json!({ "error": message }));
}

fn handle(req: &mut Request, eval: &mut TardEvaluator, info: &ServerInfo) -> Response<std::io::Cursor<Vec<u8>>>{
    let path = req.url().split('?').next().unwrap_or("").to_string();
    match (req.method(), path.as_str()){
        (Method::Get, "/") => Response::from_string(PAGE)
            .with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap()),
        (Method::Get, "/api/info") => json_response(200, info),
        (Method::Post, "/api/play") => {
            let mut body = String::new();
            if let Err(e) = req.as_reader().take(MAX_BODY_BYTES).read_to_string(&mut body){
                return error_response(400, &format!("can't read request: {}", e));
            }
            let parsed: PlayRequest = match serde_json::from_str(&body){
                Ok(p) => p,
                Err(e) => return error_response(400, &format!("bad request: {}", e)),
            };
            match play(parsed, eval, info.rollouts){
                Ok(v) => json_response(200, &v),
                Err(e) => error_response(400, &e),
            }
        }
        _ => error_response(404, "not found"),
    }
}

pub fn run(args: Args){
    let device = Device::default();
    let (tard, net) = load_model::<MiniPenteState>(&args.model, &device).unwrap_or_else(|e|{
        panic!("Failed to load model {}: {}", args.model.display(), e)
    });
    let mut eval = TardEvaluator::new(&tard);
    let info = ServerInfo{
        model: args.model.display().to_string(),
        network: format!("{} channels, {} blocks", net.channels, net.blocks),
        rollouts: args.rollouts.clamp(1, MAX_ROLLOUTS),
        max_rollouts: MAX_ROLLOUTS,
        rows: ROWS,
        cols: COLS,
    };

    let addr = format!("{}:{}", args.host, args.port);
    let server = Server::http(&addr).unwrap_or_else(|e| panic!("Can't listen on {}: {}", addr, e));
    println!("Play at http://{}/  (Ctrl-C to stop)", addr);
    // One request at a time: there's one model and one player.
    for mut req in server.incoming_requests(){
        let response = handle(&mut req, &mut eval, &info);
        if let Err(e) = req.respond(response){
            warn!("Failed to send a response: {}", e);
        }
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use tard::games::minipente::coord_2_action;

    #[test]
    fn replay_rejects_illegal_and_late_moves(){
        let a = coord_2_action(4, 4);
        assert!(replay(&[a, coord_2_action(4, 5)]).is_ok());
        assert!(replay(&[a, a]).unwrap_err().contains("move 2"));
        assert!(replay(&[81]).is_err(), "off the board");
        // Black wins with five on row 0, so a 10th move is too late.
        let win: Vec<Action> = [(0, 0), (8, 0), (0, 1), (8, 2), (0, 2), (8, 4), (0, 3), (8, 6), (0, 4), (5, 5)]
            .iter().map(|&(r, c)| coord_2_action(r, c)).collect();
        assert!(replay(&win[..9]).unwrap().is_terminal());
        assert!(replay(&win).unwrap_err().contains("after the game ended"));
    }

    #[test]
    fn view_shows_the_board_turn_and_result(){
        let moves = vec![coord_2_action(4, 4), coord_2_action(0, 8)];
        let v = view(&replay(&moves).unwrap(), moves, None);
        assert_eq!(v.board[4], "....B....");
        assert_eq!(v.board[0], "........W");
        assert_eq!(v.to_move, "Black");
        assert!(!v.legal[coord_2_action(4, 4)] && v.legal[coord_2_action(4, 5)]);
        assert!(!v.over && v.result.is_none());

        let win: Vec<Action> = [(0, 0), (8, 0), (0, 1), (8, 2), (0, 2), (8, 4), (0, 3), (8, 6), (0, 4)]
            .iter().map(|&(r, c)| coord_2_action(r, c)).collect();
        let v = view(&replay(&win).unwrap(), win, None);
        assert!(v.over);
        assert_eq!(v.result.as_deref(), Some("Black wins"));
        assert!(v.legal.iter().all(|&l| !l));
    }
}
