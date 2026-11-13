//! Hive (base game, no expansion pieces), as a training environment.
//!
//! Each player has 11 pieces: a queen, 2 spiders, 2 beetles, 3 grasshoppers
//! and 3 ants. White moves first. On a turn a player either places a piece
//! from their hand or moves one of their pieces on the board.
//! - Placing: the first piece goes anywhere; the second player's first piece
//!   must touch it. After that a new piece must touch at least one friendly
//!   piece and no enemy piece (judged by the top piece of each stack).
//! - The queen must be placed by a player's 4th turn, and (tournament rule)
//!   not on their 1st. No piece may move until its owner's queen is placed.
//! - One hive: a move may never split the pieces into two groups, not even
//!   while the piece is in transit. A piece with a beetle on it can't move.
//! - Sliding pieces (queen, spider, ant, and a beetle moving at ground level)
//!   must stay in contact with the hive, and can't squeeze through a gap
//!   between two pieces (freedom of movement).
//! - Queen: 1 step. Beetle: 1 step, and may climb onto the hive and move on
//!   top of it. Grasshopper: jumps in a straight line over one or more pieces
//!   to the first empty cell. Spider: exactly 3 steps, never revisiting a
//!   cell. Ant: any number of steps.
//! - A player whose queen is surrounded on all six sides loses; if both
//!   queens are surrounded at once it's a draw. A player with no legal move
//!   passes. As a training cutoff, a game still going after MAX_PLIES plies
//!   is a draw.
//!
//! Hive has no board, so the hive is kept centered in an N x N frame of
//! axial hex coordinates (q = column, r = row; pointy-top hexes, so row r is
//! drawn shifted right by r half-cells). The frame is re-centered after
//! every move, and actions refer to cells of the current frame.
//!
//! Text moves use the standard notation: a piece name like "wQ", "bS1" or
//! "wA3" (color, kind, number in placement order), then a position next to
//! a reference piece: "wA1 bQ-" (east of bQ), "-bQ" (west), "bQ/"
//! (north-east), "/bQ" (south-west), "\bQ" (north-west), "bQ\" (south-east),
//! or just "bQ" for a beetle climbing on top of it. The very first move is
//! just a piece name, and "pass" passes.

use std::fmt;
use tracing::trace;

use crate::games::{Action, GameState, TextGame};
use crate::model::NetConfig;

/// Side length of the square frame. 22 pieces span at most 22 cells in any
/// direction and every legal destination is next to the hive, so a
/// centered hive and all its moves always fit in 24.
pub const N: usize = 24;
pub const CELLS: usize = N * N;
const CENTER: usize = (N / 2) * N + N / 2;
/// Pieces per player.
pub const PIECES: usize = 11;
/// Action `piece * CELLS + cell` moves (or places) that piece of the player
/// to move onto that cell. The last action passes.
pub const PASS: Action = PIECES * CELLS;
/// A game still going after this many plies is a draw.
pub const MAX_PLIES: usize = 200;
/// Tournament rule: no player may place their queen on their first turn.
const NO_QUEEN_ON_FIRST_TURN: bool = true;
/// Tallest possible stack: a piece with all four beetles on top of it.
const MAX_HEIGHT: usize = 5;

const WHITE: usize = 0;
const BLACK: usize = 1;
const COLOR_CHAR: [char; 2] = ['w', 'b'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bug{
    Queen,
    Spider,
    Beetle,
    Grasshopper,
    Ant,
}

const BUGS: [Bug; 5] = [Bug::Queen, Bug::Spider, Bug::Beetle, Bug::Grasshopper, Bug::Ant];

/// Each player's pieces by index. Pieces of one kind are numbered (1, 2, 3)
/// in this order, and are always placed lowest number first.
const KIND: [Bug; PIECES] = [
    Bug::Queen,
    Bug::Spider, Bug::Spider,
    Bug::Beetle, Bug::Beetle,
    Bug::Grasshopper, Bug::Grasshopper, Bug::Grasshopper,
    Bug::Ant, Bug::Ant, Bug::Ant,
];
const QUEEN: usize = 0;

impl Bug{
    fn letter(self) -> char{
        match self{
            Bug::Queen => 'Q',
            Bug::Spider => 'S',
            Bug::Beetle => 'B',
            Bug::Grasshopper => 'G',
            Bug::Ant => 'A',
        }
    }

    fn from_letter(c: char) -> Option<Bug>{
        return BUGS.iter().copied().find(|b| b.letter() == c);
    }

    // Index of this kind's first piece in KIND.
    fn first(self) -> usize{
        return KIND.iter().position(|&k| k == self).unwrap();
    }

    // How many pieces of this kind each player has.
    fn count(self) -> usize{
        return KIND.iter().filter(|&&k| k == self).count();
    }
}

// The six neighbor directions as axial (dq, dr), counterclockwise from east:
// E, NE, NW, W, SW, SE. Directions d-1 and d+1 (mod 6) flank direction d.
const DIRS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

fn cell_qr(cell: usize) -> (i32, i32){
    return ((cell % N) as i32, (cell / N) as i32);
}

fn qr_cell(q: i32, r: i32) -> Option<usize>{
    if q < 0 || r < 0 || q >= N as i32 || r >= N as i32{
        return None;
    }
    return Some(r as usize * N + q as usize);
}

// The neighbor of `cell` in direction d, or None if it's outside the frame.
fn neighbor(cell: usize, d: usize) -> Option<usize>{
    let (q, r) = cell_qr(cell);
    return qr_cell(q + DIRS[d].0, r + DIRS[d].1);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Loc{
    cell: u16,
    level: u8, // 0 on the ground, 1+ on top of other pieces
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status{
    Ongoing,
    Won(usize), // the winning player
    Draw,
}

#[derive(Clone, Debug)]
pub struct HiveState{
    pos_: [[Option<Loc>; PIECES]; 2], // [player][piece]; None = in hand. Player 0 is White.
    ply_: usize,                      // plies played, passes included; White moves on even plies
    status_: Status,
}

// The stacks on the board, derived from the piece positions.
struct Grid{
    height: [u8; CELLS],
    stack: [[u8; MAX_HEIGHT]; CELLS], // bottom first; entries are player * PIECES + piece
}

impl Grid{
    // (player, piece) on top of the stack at `cell`, if any.
    fn top(&self, cell: usize) -> Option<(usize, usize)>{
        let h = self.height[cell] as usize;
        if h == 0{
            return None;
        }
        let id = self.stack[cell][h - 1] as usize;
        return Some((id / PIECES, id % PIECES));
    }
}

/// Default network size for a fresh Hive model. Not tuned.
pub fn default_net_config() -> NetConfig{
    return NetConfig{ channels: 64, blocks: 8, value_hidden: 64 };
}

impl Default for HiveState{
    fn default() -> Self{
        return Self::new();
    }
}

// Input planes, from the side to move's perspective ("my" = side to move):
const PLANE_MY_PIECES: usize = 0;       // 11 planes: 1 where my piece k is
const PLANE_THEIR_PIECES: usize = 11;   // 11 planes: same for the opponent
const PLANE_MY_TOP: usize = 22;         // 1 where the top of a stack is mine
const PLANE_THEIR_TOP: usize = 23;      // 1 where it's the opponent's
const PLANE_HEIGHT: usize = 24;         // stack height
const PLANE_MY_HAND: usize = 25;        // 5 planes: how many of each kind I hold
const PLANE_THEIR_HAND: usize = 30;     // 5 planes: same for the opponent
const PLANE_MY_TURN: usize = 35;        // my turn number (capped at 4) / 4, for the queen rule
const PLANE_PLY: usize = 36;            // ply / MAX_PLIES, for the draw cutoff

impl GameState for HiveState{
    const NAME: &'static str = "hive";
    const ACTN_SPC_SIZE: usize = PIECES * CELLS + 1;
    const N_CHAN: usize = 37;
    const ROWS: usize = N;
    const COLS: usize = N;

    fn to_tensor(&self) -> Vec<f32>{
        let me = self.to_move();
        let them = 1 - me;
        let mut t = vec![0.0; Self::N_CHAN * CELLS];
        let mut fill = |plane: usize, v: f32| t[plane * CELLS..(plane + 1) * CELLS].fill(v);
        for (base, p) in [(PLANE_MY_HAND, me), (PLANE_THEIR_HAND, them)]{
            for (b, bug) in BUGS.iter().enumerate(){
                fill(base + b, self.in_hand(p, *bug) as f32);
            }
        }
        fill(PLANE_MY_TURN, (self.ply_ / 2).min(4) as f32 / 4.0);
        fill(PLANE_PLY, self.ply_ as f32 / MAX_PLIES as f32);
        for (base, p) in [(PLANE_MY_PIECES, me), (PLANE_THEIR_PIECES, them)]{
            for i in 0..PIECES{
                if let Some(loc) = self.pos_[p][i]{
                    t[(base + i) * CELLS + loc.cell as usize] = 1.0;
                }
            }
        }
        let g = self.grid();
        for cell in 0..CELLS{
            if let Some((p, _)) = g.top(cell){
                let plane = if p == me {PLANE_MY_TOP} else {PLANE_THEIR_TOP};
                t[plane * CELLS + cell] = 1.0;
            }
            t[PLANE_HEIGHT * CELLS + cell] = g.height[cell] as f32;
        }
        return t;
    }

    fn legal_actions_mask(&self) -> Vec<bool>{
        let mut mask = vec![false; Self::ACTN_SPC_SIZE];
        for a in self.legal_actions(){
            mask[a] = true;
        }
        return mask;
    }

    fn apply_action(&mut self, action: Action){
        trace!("Applying action {}", action);
        assert!(!self.is_terminal(), "apply_action({}) called on a finished game\n{}", action, self);
        assert!(self.is_legal_action(action), "Illegal action {} for board\n{}", action, self);
        if action != PASS{
            let (piece, cell) = (action / CELLS, action % CELLS);
            // The piece leaves its old cell (if any), which is never `cell`.
            let level = self.grid().height[cell];
            let me = self.to_move();
            self.pos_[me][piece] = Some(Loc{ cell: cell as u16, level });
        }
        self.ply_ += 1;
        self.update_status();
        self.recenter();
    }

    fn is_terminal(&self) -> bool{
        return self.status_ != Status::Ongoing;
    }

    // Unlike Pente, a player can lose on their own move (by surrounding their
    // own queen), so this can be +1.
    fn get_terminal_value(&self) -> f32{
        match self.status_{
            Status::Won(p) => if p == self.to_move() {1.0} else {-1.0},
            Status::Draw => 0.0,
            Status::Ongoing => panic!("get_terminal_value called on a non-terminal state\n{}", self),
        }
    }
}


impl HiveState{
    pub fn new() -> Self{
        return Self{
            pos_: [[None; PIECES]; 2],
            ply_: 0,
            status_: Status::Ongoing,
        };
    }

    fn to_move(&self) -> usize{
        return self.ply_ % 2;
    }

    fn in_hand(&self, player: usize, bug: Bug) -> usize{
        let first = bug.first();
        return (first..first + bug.count()).filter(|&i| self.pos_[player][i].is_none()).count();
    }

    fn on_board(&self, player: usize) -> bool{
        return self.pos_[player].iter().any(|p| p.is_some());
    }

    fn grid(&self) -> Grid{
        let mut g = Grid{ height: [0; CELLS], stack: [[0; MAX_HEIGHT]; CELLS] };
        for p in 0..2{
            for i in 0..PIECES{
                if let Some(loc) = self.pos_[p][i]{
                    let (cell, level) = (loc.cell as usize, loc.level as usize);
                    g.stack[cell][level] = (p * PIECES + i) as u8;
                    g.height[cell] = g.height[cell].max(level as u8 + 1);
                }
            }
        }
        return g;
    }

    /// All legal actions. Never empty for an ongoing game: PASS when nothing
    /// else is legal.
    pub fn legal_actions(&self) -> Vec<Action>{
        if self.is_terminal(){
            return vec![];
        }
        let me = self.to_move();
        let g = self.grid();
        let mut actions = Vec::new();
        let cells = self.placement_cells(&g);
        for i in 0..PIECES{
            if self.placeable(i){
                actions.extend(cells.iter().map(|&c| i * CELLS + c));
            }
        }
        if self.pos_[me][QUEEN].is_some(){
            for i in 0..PIECES{
                actions.extend(self.piece_moves(&g, me, i).into_iter().map(|c| i * CELLS + c));
            }
        }
        if actions.is_empty(){
            actions.push(PASS);
        }
        return actions;
    }

    pub fn is_legal_action(&self, action: Action) -> bool{
        if self.is_terminal() || action > PASS{
            return false;
        }
        if action == PASS{
            return self.legal_actions() == [PASS];
        }
        let (piece, cell) = (action / CELLS, action % CELLS);
        let me = self.to_move();
        let g = self.grid();
        if self.pos_[me][piece].is_none(){
            return self.placeable(piece) && self.placement_ok(&g, cell);
        }
        return self.pos_[me][QUEEN].is_some() && self.piece_moves(&g, me, piece).contains(&cell);
    }

    // Can the player to move place piece i this turn (wherever)?
    fn placeable(&self, i: usize) -> bool{
        let me = self.to_move();
        let bug = KIND[i];
        if self.pos_[me][i].is_some(){
            return false;
        }
        // Lower-numbered pieces of the same kind go first.
        if (bug.first()..i).any(|j| self.pos_[me][j].is_none()){
            return false;
        }
        let turn = self.ply_ / 2;
        if bug == Bug::Queen && turn == 0 && NO_QUEEN_ON_FIRST_TURN{
            return false;
        }
        // The queen must be down by the 4th turn.
        if turn >= 3 && self.pos_[me][QUEEN].is_none() && bug != Bug::Queen{
            return false;
        }
        return true;
    }

    // Could the player to move place a piece on `cell`?
    fn placement_ok(&self, g: &Grid, cell: usize) -> bool{
        if g.height[cell] != 0{
            return false;
        }
        let me = self.to_move();
        if !self.on_board(WHITE) && !self.on_board(BLACK){
            return cell == CENTER; // the first piece: every cell is the same
        }
        let (mut friend, mut foe) = (false, false);
        for d in 0..6{
            if let Some((p, _)) = neighbor(cell, d).and_then(|n| g.top(n)){
                if p == me {friend = true} else {foe = true}
            }
        }
        if !self.on_board(me){
            return friend || foe; // a player's first piece just has to touch the hive
        }
        return friend && !foe;
    }

    fn placement_cells(&self, g: &Grid) -> Vec<usize>{
        if !self.on_board(WHITE) && !self.on_board(BLACK){
            return vec![CENTER];
        }
        let mut seen = [false; CELLS];
        let mut cells = Vec::new();
        for cell in 0..CELLS{
            if g.height[cell] == 0{
                continue;
            }
            for d in 0..6{
                if let Some(n) = neighbor(cell, d){
                    if !seen[n]{
                        seen[n] = true;
                        if self.placement_ok(g, n){
                            cells.push(n);
                        }
                    }
                }
            }
        }
        return cells;
    }

    // Cells player p's piece i (on the board) can move to, ignoring whether
    // p's queen is placed.
    fn piece_moves(&self, g: &Grid, p: usize, i: usize) -> Vec<usize>{
        let Some(loc) = self.pos_[p][i] else{
            return vec![];
        };
        let from = loc.cell as usize;
        if g.height[from] != loc.level + 1{
            return vec![]; // something is on top of it
        }
        // Heights with the piece lifted off.
        let mut h = g.height;
        h[from] -= 1;
        if h[from] == 0 && !connected(&h){
            return vec![]; // moving it would split the hive
        }
        match KIND[i]{
            Bug::Queen => (0..6).filter_map(|d|{
                let to = neighbor(from, d)?;
                (h[to] == 0 && can_step(&h, from, d, 0, 0)).then_some(to)
            }).collect(),
            Bug::Beetle => (0..6).filter_map(|d|{
                let to = neighbor(from, d)?;
                can_step(&h, from, d, h[from], h[to]).then_some(to)
            }).collect(),
            Bug::Grasshopper => (0..6).filter_map(|d|{
                let mut to = neighbor(from, d)?;
                if h[to] == 0{
                    return None; // must jump over at least one piece
                }
                while h[to] > 0{
                    to = neighbor(to, d)?;
                }
                Some(to)
            }).collect(),
            Bug::Spider => spider_moves(&h, from),
            Bug::Ant => ant_moves(&h, from),
        }
    }

    // After a move: a surrounded queen loses; both surrounded is a draw; a
    // game at the ply limit is a draw.
    fn update_status(&mut self){
        let g = self.grid();
        let surrounded = |p: usize| match self.pos_[p][QUEEN]{
            Some(loc) => (0..6).all(|d| neighbor(loc.cell as usize, d).is_some_and(|n| g.height[n] > 0)),
            None => false,
        };
        self.status_ = match (surrounded(WHITE), surrounded(BLACK)){
            (true, true) => Status::Draw,
            (true, false) => Status::Won(BLACK),
            (false, true) => Status::Won(WHITE),
            (false, false) if self.ply_ >= MAX_PLIES => Status::Draw,
            (false, false) => Status::Ongoing,
        };
    }

    // Shifts every piece so the hive's bounding box sits in the middle of
    // the frame.
    fn recenter(&mut self){
        let cells: Vec<(i32, i32)> = self.pos_.iter().flatten().flatten()
            .map(|loc| cell_qr(loc.cell as usize)).collect();
        if cells.is_empty(){
            return;
        }
        let (qmin, qmax) = (cells.iter().map(|c| c.0).min().unwrap(), cells.iter().map(|c| c.0).max().unwrap());
        let (rmin, rmax) = (cells.iter().map(|c| c.1).min().unwrap(), cells.iter().map(|c| c.1).max().unwrap());
        let dq = (N as i32 - (qmax - qmin + 1)) / 2 - qmin;
        let dr = (N as i32 - (rmax - rmin + 1)) / 2 - rmin;
        if dq == 0 && dr == 0{
            return;
        }
        for loc in self.pos_.iter_mut().flatten().flatten(){
            let (q, r) = cell_qr(loc.cell as usize);
            loc.cell = qr_cell(q + dq, r + dr).expect("hive doesn't fit in the frame") as u16;
        }
    }

    // e.g. "wQ", "bA2"
    fn piece_name(player: usize, i: usize) -> String{
        let bug = KIND[i];
        if bug == Bug::Queen{
            return format!("{}Q", COLOR_CHAR[player]);
        }
        return format!("{}{}{}", COLOR_CHAR[player], bug.letter(), i - bug.first() + 1);
    }

    fn parse_piece(name: &str) -> Option<(usize, usize)>{
        let mut chars = name.chars();
        let color = chars.next()?;
        let player = COLOR_CHAR.iter().position(|&c| c == color)?;
        let bug = Bug::from_letter(chars.next()?)?;
        let rest = chars.as_str();
        let number = if bug == Bug::Queen{
            if !rest.is_empty(){
                return None;
            }
            1
        }else{
            rest.parse::<usize>().ok()?
        };
        if number == 0 || number > bug.count(){
            return None;
        }
        return Some((player, bug.first() + number - 1));
    }

    // Parses a position like "bQ-", "/wA1" or "wQ" into a cell.
    fn parse_position(&self, text: &str) -> Result<usize, String>{
        const MARKS: [char; 3] = ['-', '/', '\\'];
        let first = text.chars().next().filter(|c| MARKS.contains(c));
        let last = text.chars().last().filter(|c| MARKS.contains(c));
        let name = text.trim_start_matches(MARKS).trim_end_matches(MARKS);
        let (p, i) = Self::parse_piece(name).ok_or(format!("unknown piece '{}'", name))?;
        let Some(loc) = self.pos_[p][i] else{
            return Err(format!("{} isn't on the board", name));
        };
        let d = match (first, last){
            (None, None) => return Ok(loc.cell as usize), // on top of it
            (None, Some('-')) => 0,
            (None, Some('/')) => 1,
            (Some('\\'), None) => 2,
            (Some('-'), None) => 3,
            (Some('/'), None) => 4,
            (None, Some('\\')) => 5,
            _ => return Err(format!("bad position '{}'", text)),
        };
        return neighbor(loc.cell as usize, d).ok_or(format!("'{}' is outside the frame", text));
    }
}

// Is every occupied cell connected to every other?
fn connected(h: &[u8; CELLS]) -> bool{
    let occupied = h.iter().filter(|&&x| x > 0).count();
    let Some(start) = h.iter().position(|&x| x > 0) else{
        return true;
    };
    let mut seen = [false; CELLS];
    seen[start] = true;
    let mut stack = vec![start];
    let mut reached = 1;
    while let Some(c) = stack.pop(){
        for d in 0..6{
            if let Some(n) = neighbor(c, d){
                if h[n] > 0 && !seen[n]{
                    seen[n] = true;
                    reached += 1;
                    stack.push(n);
                }
            }
        }
    }
    return reached == occupied;
}

// Can a piece resting on a stack of height `ha` at `from` move one cell in
// direction d onto a stack of height `hb`? `h` has the moving piece lifted
// off. It can't pass between two neighbors that are both higher than its
// path (freedom of movement), and at ground level one of them must be
// occupied so it keeps touching the hive.
fn can_step(h: &[u8; CELLS], from: usize, d: usize, ha: u8, hb: u8) -> bool{
    let h1 = neighbor(from, (d + 5) % 6).map_or(0, |c| h[c]);
    let h2 = neighbor(from, (d + 1) % 6).map_or(0, |c| h[c]);
    let path = ha.max(hb);
    if h1 > path && h2 > path{
        return false;
    }
    return path > 0 || h1 > 0 || h2 > 0;
}

// Ground cells reachable from `from` by exactly 3 slides, never revisiting a cell.
fn spider_moves(h: &[u8; CELLS], from: usize) -> Vec<usize>{
    fn walk(h: &[u8; CELLS], path: &mut Vec<usize>, out: &mut Vec<usize>){
        let cur = *path.last().unwrap();
        if path.len() == 4{
            if !out.contains(&cur){
                out.push(cur);
            }
            return;
        }
        for d in 0..6{
            if let Some(n) = neighbor(cur, d){
                if h[n] == 0 && !path.contains(&n) && can_step(h, cur, d, 0, 0){
                    path.push(n);
                    walk(h, path, out);
                    path.pop();
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(h, &mut vec![from], &mut out);
    return out;
}

// Ground cells reachable from `from` by any number of slides.
fn ant_moves(h: &[u8; CELLS], from: usize) -> Vec<usize>{
    let mut seen = [false; CELLS];
    seen[from] = true;
    let mut stack = vec![from];
    let mut out = Vec::new();
    while let Some(c) = stack.pop(){
        for d in 0..6{
            if let Some(n) = neighbor(c, d){
                if h[n] == 0 && !seen[n] && can_step(h, c, d, 0, 0){
                    seen[n] = true;
                    out.push(n);
                    stack.push(n);
                }
            }
        }
    }
    return out;
}


impl TextGame for HiveState{
    const PLAYERS: [&'static str; 2] = ["White", "Black"];
    const MOVE_FORMAT: &'static str = r"e.g. wA1, bS1 wA1-, wG1 -bQ, wB1 bQ (on top), wQ /bA2, bA1 wS1\, pass";

    fn active_player(&self) -> &'static str{
        return Self::PLAYERS[self.to_move()];
    }

    fn parse_action(&self, input: &str) -> Result<Action, String>{
        let words: Vec<&str> = input.split_whitespace().collect();
        let action = match words.as_slice(){
            [] => return Err("no move given".to_string()),
            ["pass"] => PASS,
            [piece, rest @ ..] if rest.len() <= 1 => {
                let (p, i) = Self::parse_piece(piece).ok_or(format!("unknown piece '{}'", piece))?;
                if p != self.to_move(){
                    return Err(format!("{} is not {}'s piece", piece, self.active_player()));
                }
                let cell = match rest{
                    [position] => self.parse_position(position)?,
                    _ if !self.on_board(WHITE) && !self.on_board(BLACK) => CENTER,
                    _ => return Err(format!("say where {} goes, e.g. '{} {}'", piece, piece,
                                            Self::piece_name(1 - p, QUEEN) + "-")),
                };
                i * CELLS + cell
            }
            _ => return Err("too many words".to_string()),
        };
        if !self.is_legal_action(action){
            return Err(format!("'{}' is not a legal move", input.trim()));
        }
        return Ok(action);
    }

    fn action_text(&self, action: Action) -> String{
        if action == PASS{
            return "pass".to_string();
        }
        let me = self.to_move();
        let (piece, cell) = (action / CELLS, action % CELLS);
        let name = Self::piece_name(me, piece);
        if !self.on_board(WHITE) && !self.on_board(BLACK){
            return name;
        }
        let g = self.grid();
        let mover = (me * PIECES + piece) as u8;
        // The piece that will be directly below or beside the mover:
        // the top of a stack, unless that's the mover itself.
        let reference = |c: usize| -> Option<String>{
            let h = g.height[c] as usize;
            let mut level = h.checked_sub(1)?;
            if g.stack[c][level] == mover{
                level = level.checked_sub(1)?;
            }
            let id = g.stack[c][level] as usize;
            return Some(Self::piece_name(id / PIECES, id % PIECES));
        };
        if let Some(below) = reference(cell){
            return format!("{} {}", name, below);
        }
        for d in 0..6{
            // The reference sits in direction d+3 from the destination, so
            // the destination is in direction d from it.
            let Some(r) = neighbor(cell, (d + 3) % 6).and_then(|c| reference(c)) else{
                continue;
            };
            let position = match d{
                0 => format!("{}-", r),
                1 => format!("{}/", r),
                2 => format!("\\{}", r),
                3 => format!("-{}", r),
                4 => format!("/{}", r),
                _ => format!("{}\\", r),
            };
            return format!("{} {}", name, position);
        }
        unreachable!("action {} has no neighboring piece to describe it by", action);
    }
}


impl fmt::Display for HiveState{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result{
        for p in [WHITE, BLACK]{
            let hand: Vec<String> = (0..PIECES).filter(|&i| self.pos_[p][i].is_none())
                .map(|i| Self::piece_name(p, i)).collect();
            writeln!(f, "{} in hand: {}", Self::PLAYERS[p], if hand.is_empty() {"-".to_string()} else {hand.join(" ")})?;
        }
        let g = self.grid();
        let occupied: Vec<usize> = (0..CELLS).filter(|&c| g.height[c] > 0).collect();
        if occupied.is_empty(){
            return writeln!(f, "(empty board)");
        }
        // Row r is shifted right by r half-cells; x = 2q + r is a cell's
        // column in half-cells.
        let qr: Vec<(i32, i32)> = occupied.iter().map(|&c| cell_qr(c)).collect();
        let (rmin, rmax) = (qr.iter().map(|c| c.1).min().unwrap() - 1, qr.iter().map(|c| c.1).max().unwrap() + 1);
        let (xmin, xmax) = (qr.iter().map(|c| 2 * c.0 + c.1).min().unwrap() - 2,
                            qr.iter().map(|c| 2 * c.0 + c.1).max().unwrap() + 2);
        for r in rmin..=rmax{
            let mut line = String::new();
            let mut x = xmin;
            if (x - r).rem_euclid(2) == 1{
                line.push_str("  ");
                x += 1;
            }
            while x <= xmax{
                let label = match qr_cell((x - r) / 2, r).and_then(|c| g.top(c)){
                    Some((p, i)) => Self::piece_name(p, i),
                    None => ".".to_string(),
                };
                line.push_str(&format!("{:<4}", label));
                x += 2;
            }
            writeln!(f, "{}", line.trim_end())?;
        }
        for &c in &occupied{
            let h = g.height[c] as usize;
            if h > 1{
                let names: Vec<String> = (0..h).rev().map(|l|{
                    let id = g.stack[c][l] as usize;
                    Self::piece_name(id / PIECES, id % PIECES)
                }).collect();
                writeln!(f, "stack (top first): {}", names.join(" on "))?;
            }
        }
        match self.status_{
            Status::Ongoing => writeln!(f, "{} to move (ply {})", self.active_player(), self.ply_),
            Status::Won(p) => writeln!(f, "{} wins", Self::PLAYERS[p]),
            Status::Draw => writeln!(f, "Draw"),
        }
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand::rngs::StdRng;

    // A position from (player, piece, q, r) placements around the frame
    // center, all on the ground, with `ply` plies played. Not re-centered.
    fn position(pieces: &[(usize, usize, i32, i32)], ply: usize) -> HiveState{
        let mut s = HiveState::new();
        for &(p, i, q, r) in pieces{
            put(&mut s, p, i, q, r, 0);
        }
        s.ply_ = ply;
        return s;
    }

    fn put(s: &mut HiveState, p: usize, i: usize, q: i32, r: i32, level: u8){
        s.pos_[p][i] = Some(Loc{ cell: at(q, r) as u16, level });
    }

    // The cell (q, r) away from the frame center.
    fn at(q: i32, r: i32) -> usize{
        let (cq, cr) = cell_qr(CENTER);
        return qr_cell(cq + q, cr + r).unwrap();
    }

    fn moves(s: &HiveState, i: usize) -> Vec<usize>{
        let mut m = s.piece_moves(&s.grid(), s.to_move(), i);
        m.sort();
        return m;
    }

    fn cells(list: &[(i32, i32)]) -> Vec<usize>{
        let mut v: Vec<usize> = list.iter().map(|&(q, r)| at(q, r)).collect();
        v.sort();
        return v;
    }

    fn play(s: &mut HiveState, text: &str){
        let a = s.parse_action(text).unwrap_or_else(|e| panic!("{}: {}\n{}", text, e, s));
        s.apply_action(a);
    }

    // Piece indices.
    const S1: usize = 1;
    const S2: usize = 2;
    const B1: usize = 3;
    const G1: usize = 5;
    const G2: usize = 6;
    const A1: usize = 8;
    const A2: usize = 9;
    const A3: usize = 10;

    #[test]
    fn piece_kinds_and_names(){
        assert_eq!(BUGS.map(|b| b.count()), [1, 2, 2, 3, 3]);
        assert_eq!(HiveState::piece_name(WHITE, QUEEN), "wQ");
        assert_eq!(HiveState::piece_name(BLACK, A3), "bA3");
        for p in [WHITE, BLACK]{
            for i in 0..PIECES{
                assert_eq!(HiveState::parse_piece(&HiveState::piece_name(p, i)), Some((p, i)));
            }
        }
        for bad in ["wQ1", "wA4", "wA0", "xA1", "wX1", "wA", "w"]{
            assert_eq!(HiveState::parse_piece(bad), None, "{bad}");
        }
    }

    #[test]
    fn first_move_is_any_piece_but_the_queen_at_the_center(){
        let s = HiveState::new();
        assert_eq!(s.active_player(), "White");
        let mut legal = s.legal_actions();
        legal.sort();
        assert_eq!(legal, vec![S1 * CELLS + CENTER, B1 * CELLS + CENTER, G1 * CELLS + CENTER, A1 * CELLS + CENTER]);
        assert!(s.parse_action("wQ").is_err(), "no queen on the first turn");
        assert!(s.parse_action("wS2").is_err(), "spiders are placed S1 first");
        assert!(s.parse_action("bA1").is_err(), "White moves first");
    }

    #[test]
    fn blacks_first_piece_may_touch_white_and_later_pieces_may_not(){
        let mut s = HiveState::new();
        play(&mut s, "wA1");
        // 6 cells around wA1, for each of S, B, G, A (still no queen).
        assert_eq!(s.legal_actions().len(), 6 * 4);
        play(&mut s, "bA1 wA1-");
        // White's second piece: next to wA1 but not next to bA1, so 3 cells,
        // for each of Q, S, B, G, A.
        let legal = s.legal_actions();
        assert_eq!(legal.len(), 3 * 5);
        assert!(s.parse_action("wQ wA1/").is_err(), "north-east of wA1 touches bA1");
        assert!(s.parse_action("wQ wA1\\").is_err(), "south-east of wA1 touches bA1");
        play(&mut s, "wQ -wA1");
    }

    #[test]
    fn queen_must_be_placed_by_the_fourth_turn_and_before_moving(){
        let mut s = HiveState::new();
        for m in ["wA1", "bA1 wA1-", "wA2 -wA1", "bA2 bA1-", "wA3 -wA2"]{
            play(&mut s, m);
        }
        // No queen yet, so Black's pieces can't move: only placements.
        let me = s.to_move();
        assert!(s.legal_actions().iter().all(|&a| s.pos_[me][a / CELLS].is_none()));
        play(&mut s, "bA3 bA2-");
        // White's 4th turn without the queen: only the queen may be placed.
        let legal = s.legal_actions();
        assert!(!legal.is_empty());
        assert!(legal.iter().all(|&a| a / CELLS == QUEEN), "{:?}", legal);
        play(&mut s, "wQ -wA3");
        // Now White's pieces can move (the queen, at the end of the line).
        play(&mut s, "bQ bA3-");
        assert!(s.legal_actions().iter().any(|&a| a != PASS && s.pos_[WHITE][a / CELLS].is_some()));
    }

    #[test]
    fn grasshopper_jumps_over_a_line_to_the_first_gap(){
        // wG1 - bQ - bA1 in a row east, wQ hanging off bQ to the south-east.
        let s = position(&[(WHITE, G1, 0, 0), (BLACK, QUEEN, 1, 0), (BLACK, A1, 2, 0), (WHITE, QUEEN, 1, 1)], 10);
        assert_eq!(moves(&s, G1), cells(&[(3, 0)]));
    }

    #[test]
    fn a_piece_holding_the_hive_together_cannot_move(){
        let s = position(&[(WHITE, QUEEN, 0, 0), (WHITE, A1, 1, 0), (BLACK, QUEEN, 2, 0)], 10);
        assert!(moves(&s, A1).is_empty());
        assert!(!moves(&s, QUEEN).is_empty());
    }

    #[test]
    fn ant_cannot_squeeze_into_a_gated_pocket(){
        // (0, 0) is enclosed on five sides; its only open side, (0, 1), is
        // flanked by (1, 0) and (-1, 1), too narrow to slide through.
        let s = position(&[
            (WHITE, QUEEN, 1, 0), (BLACK, QUEEN, 1, -1), (BLACK, S1, 0, -1),
            (BLACK, S2, -1, 0), (BLACK, B1, -1, 1), (WHITE, A1, 2, 0),
        ], 10);
        let m = moves(&s, A1);
        assert!(!m.contains(&at(0, 0)), "slid into the pocket");
        assert!(m.contains(&at(0, 1)), "the pocket's mouth is reachable");
        assert!(!m.contains(&at(2, 0)), "the ant can't stay put");
        // Every ant destination touches the hive.
        for &c in &m{
            assert!((0..6).any(|d| neighbor(c, d).is_some_and(|n| n != at(2, 0) && s.grid().height[n] > 0)));
        }
    }

    #[test]
    fn spider_moves_exactly_three_cells_around_the_hive(){
        // The hive is a domino; the spider walks its ring three steps either way.
        let s = position(&[(WHITE, QUEEN, 0, 0), (BLACK, QUEEN, 1, 0), (WHITE, S1, -1, 0)], 10);
        assert_eq!(moves(&s, S1), cells(&[(2, -1), (1, 1)]));
    }

    #[test]
    fn queen_slides_one_cell_keeping_contact(){
        let s = position(&[(WHITE, QUEEN, 0, 0), (BLACK, QUEEN, 1, 0)], 10);
        assert_eq!(moves(&s, QUEEN), cells(&[(1, -1), (0, 1)]));
    }

    #[test]
    fn beetle_climbs_and_walks_on_top(){
        let mut s = position(&[(WHITE, QUEEN, 0, 0), (BLACK, QUEEN, 1, 0), (WHITE, B1, -1, 0)], 10);
        assert_eq!(moves(&s, B1), cells(&[(0, 0), (0, -1), (-1, 1)]));
        play(&mut s, "wB1 wQ");
        assert_eq!(s.pos_[WHITE][B1], Some(Loc{ cell: s.pos_[WHITE][QUEEN].unwrap().cell, level: 1 }));
        s.ply_ += 1; // White to move again
        assert!(moves(&s, QUEEN).is_empty(), "a covered piece can't move");
        assert_eq!(moves(&s, B1).len(), 6, "from on top it can step down anywhere");
        assert!(s.parse_action("wB1 bQ").is_ok(), "or onto the next stack");
    }

    #[test]
    fn beetle_cannot_pass_between_two_taller_stacks(){
        // A ring around (0, 0) open only at (1, 0). The way east from (0, 0)
        // to (1, 0) runs between (1, -1) and (0, 1), both stacks of height 2.
        let mut s = position(&[
            (WHITE, QUEEN, 0, -1), (WHITE, B1, 0, 0),
            (BLACK, QUEEN, 1, -1), (BLACK, A1, 0, 1), (BLACK, A2, -1, 0), (BLACK, A3, -1, 1),
        ], 10);
        put(&mut s, BLACK, B1, 1, -1, 1);
        put(&mut s, BLACK, B1 + 1, 0, 1, 1);
        let m = moves(&s, B1);
        assert!(!m.is_empty(), "the beetle can still climb onto its neighbors");
        assert!(!m.contains(&at(1, 0)), "squeezed between two taller stacks");
        // From on top of a 2-high stack it passes over them.
        put(&mut s, WHITE, A1, 0, 0, 0);
        put(&mut s, WHITE, B1 + 1, 0, 0, 1);
        put(&mut s, WHITE, B1, 0, 0, 2);
        assert!(moves(&s, B1).contains(&at(1, 0)));
    }

    #[test]
    fn placement_goes_by_the_top_piece_and_no_move_means_pass(){
        // Black's beetle sits on White's queen: White touches no white piece,
        // can't place anything, and its only piece is covered.
        let mut s = position(&[(WHITE, QUEEN, 0, 0), (BLACK, QUEEN, 1, 0)], 6);
        put(&mut s, BLACK, B1, 0, 0, 1);
        assert_eq!(s.legal_actions(), vec![PASS]);
        assert!(s.is_legal_action(PASS));
        assert_eq!(s.parse_action("pass"), Ok(PASS));
        play(&mut s, "pass");
        assert_eq!(s.active_player(), "Black");
        assert!(!s.is_legal_action(PASS), "pass only when nothing else is legal");
    }

    #[test]
    fn surrounding_a_queen_wins_and_surrounding_both_draws(){
        // bQ at (0, 0) is enclosed except at (0, 1); White's beetle steps in.
        let pieces = [
            (BLACK, QUEEN, 0, 0), (WHITE, QUEEN, 1, 0), (WHITE, B1, 1, 1),
            (BLACK, A1, 1, -1), (BLACK, A2, 0, -1), (BLACK, A3, -1, 0), (BLACK, S1, -1, 1),
        ];
        let mut s = position(&pieces, 10);
        play(&mut s, "wB1 /wQ");
        assert!(s.is_terminal());
        assert_eq!(s.status_, Status::Won(WHITE));
        assert_eq!(s.get_terminal_value(), -1.0, "Black, to move, has lost");

        // Same, but it's White's own queen in the pocket: White loses on its
        // own move, so the side to move (Black) has won.
        let mut pieces = pieces;
        pieces[0].0 = WHITE;
        pieces[1].0 = BLACK;
        let mut s = position(&pieces, 10);
        play(&mut s, "wB1 /bQ");
        assert_eq!(s.status_, Status::Won(BLACK));
        assert_eq!(s.get_terminal_value(), 1.0);

        // Both queens share the last open cell (0, 1).
        let mut s = position(&[
            (WHITE, QUEEN, 0, 0), (BLACK, QUEEN, 1, 0),
            (BLACK, A1, 1, -1), (BLACK, A2, 0, -1), (BLACK, A3, -1, 0), (BLACK, S1, -1, 1),
            (BLACK, S2, 2, 0), (BLACK, G1, 2, -1), (BLACK, G2, 1, 1), (WHITE, B1, 0, 2),
        ], 10);
        play(&mut s, "wB1 -bG2");
        assert_eq!(s.status_, Status::Draw);
        assert_eq!(s.get_terminal_value(), 0.0);
    }

    #[test]
    fn game_is_drawn_at_the_ply_limit(){
        let mut s = HiveState::new();
        s.ply_ = MAX_PLIES - 1;
        s.apply_action(s.legal_actions()[0]);
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), 0.0);
    }

    #[test]
    fn to_tensor_is_from_the_side_to_moves_perspective(){
        let mut s = HiveState::new();
        play(&mut s, "wA1");
        let t = s.to_tensor();
        assert_eq!(t.len(), HiveState::N_CHAN * CELLS);
        let cell = s.pos_[WHITE][A1].unwrap().cell as usize;
        let plane = |p: usize| &t[p * CELLS..(p + 1) * CELLS];
        assert_eq!(plane(PLANE_THEIR_PIECES + A1)[cell], 1.0, "White's ant, seen by Black");
        assert_eq!(plane(PLANE_MY_PIECES + A1)[cell], 0.0);
        assert_eq!(plane(PLANE_THEIR_TOP)[cell], 1.0);
        assert_eq!(plane(PLANE_HEIGHT)[cell], 1.0);
        assert!(plane(PLANE_MY_HAND + 4).iter().all(|&x| x == 3.0), "Black holds 3 ants");
        assert!(plane(PLANE_THEIR_HAND + 4).iter().all(|&x| x == 2.0), "White holds 2 ants");
        assert!(plane(PLANE_PLY).iter().all(|&x| x == 1.0 / MAX_PLIES as f32));
    }

    #[test]
    fn display_draws_the_hive(){
        let mut s = HiveState::new();
        assert!(s.to_string().contains("(empty board)"));
        for m in ["wG1", "bA1 wG1-", "wQ -wG1"]{
            play(&mut s, m);
        }
        let text = s.to_string();
        assert!(text.contains("wQ  wG1 bA1"), "{text}");
        assert!(text.contains("Black to move"), "{text}");
    }

    // Checks the stacks are consistent, the hive is in one piece, and it
    // sits inside the frame with room for a ring of neighbors.
    fn assert_consistent(s: &HiveState){
        let g = s.grid();
        let mut count = [0u8; CELLS];
        for loc in s.pos_.iter().flatten().flatten(){
            count[loc.cell as usize] += 1;
        }
        for c in 0..CELLS{
            assert_eq!(count[c], g.height[c], "stack at cell {c} has a gap\n{s}");
            if g.height[c] > 0{
                let (q, r) = cell_qr(c);
                assert!(q > 0 && r > 0 && q < N as i32 - 1 && r < N as i32 - 1, "piece at the frame edge\n{s}");
            }
        }
        assert!(connected(&g.height), "hive split\n{s}");
    }

    #[test]
    fn random_playouts_keep_the_hive_whole_and_moves_round_trip_as_text(){
        let mut rng = StdRng::seed_from_u64(7);
        let mut decided = 0;
        for _ in 0..30{
            let mut s = HiveState::new();
            while !s.is_terminal(){
                let legal = s.legal_actions();
                assert!(!legal.is_empty());
                let mask = s.legal_actions_mask();
                assert_eq!(mask.iter().filter(|&&l| l).count(), legal.len(), "duplicate legal actions");
                // Spot-check a few actions each ply (all of them is slow in debug builds).
                for _ in 0..4{
                    let a = legal[rng.random_range(0..legal.len())];
                    assert!(s.is_legal_action(a));
                    let text = s.action_text(a);
                    assert_eq!(s.parse_action(&text), Ok(a), "'{text}' doesn't read back\n{s}");
                }
                s.apply_action(legal[rng.random_range(0..legal.len())]);
                assert_consistent(&s);
            }
            assert!(s.legal_actions().is_empty());
            if s.get_terminal_value() != 0.0{
                decided += 1;
            }
        }
        assert!(decided > 0, "no random game ended with a surrounded queen");
    }
}
