//! Ultimate tic-tac-toe: a 9x9 grid made of nine small 3x3 boards.
//!
//! X moves first and players alternate marking an empty cell. Where you
//! play inside a small board picks the small board your opponent must play
//! in next (playing in a small board's top-right cell sends them to the
//! top-right board). If that board is already decided, they may play in any
//! undecided board. Three in a row in a small board wins it; a full small
//! board with no line is a tie. No one may play in a decided board. Three
//! won small boards in a row (across, down or diagonal) wins the game; if
//! every small board is decided without that, the game is a draw.

use std::fmt;
use tracing::trace;

use crate::games::{parse_coord, Action, GameState, TextGame, WebGame};
use crate::model::NetConfig;

pub const ROWS: usize = 9;
pub const COLS: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark{
    Empty,
    X,
    O,
}

// A small board, or the whole game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status{
    Open,
    Won(Mark),
    Tied,
}

// The eight lines of a 3x3 board, by cell index (row * 3 + col).
const LINES: [[usize; 3]; 8] = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6],            // diagonals
];

#[derive(Clone, Debug)]
pub struct UtttState{
    cells_: [[Mark; COLS]; ROWS],
    boards_: [Status; 9], // each small board, numbered like cells within a board
    next_: Option<usize>, // the small board the player to move must play in; None = any open one
    x_to_move_: bool,
    status_: Status,      // the whole game; Tied means drawn
}

/// Default network size for a fresh model: as for MiniPente, the other 9x9
/// game. Not tuned.
pub fn default_net_config() -> NetConfig{
    return NetConfig{ channels: 64, blocks: 6, value_hidden: 64 };
}

impl Default for UtttState{
    fn default() -> Self{
        return Self::new();
    }
}

// Input planes, from the side to move's perspective.
// No plane says where the side to move may play: the network has to work
// that out itself.
const PLANE_MINE: usize = 0;       // my marks
const PLANE_THEIRS: usize = 1;     // the opponent's marks
const PLANE_WON_MINE: usize = 2;   // small boards I've won (the whole 3x3 block)
const PLANE_WON_THEIRS: usize = 3; // small boards the opponent has won
const PLANE_TIED: usize = 4;       // tied small boards

impl GameState for UtttState{
    const NAME: &'static str = "uttt";
    const ACTN_SPC_SIZE: usize = ROWS * COLS;
    const N_CHAN: usize = 5;
    const ROWS: usize = ROWS;
    const COLS: usize = COLS;

    fn to_tensor(&self) -> Vec<f32>{
        let plane = ROWS * COLS;
        let mut t = vec![0.0; Self::N_CHAN * plane];
        let (mine, theirs) = (self.active_mark(), self.inactive_mark());
        for row in 0..ROWS{
            for col in 0..COLS{
                let i = row * COLS + col;
                let cell = self.cells_[row][col];
                if cell == mine{
                    t[PLANE_MINE * plane + i] = 1.0;
                }else if cell == theirs{
                    t[PLANE_THEIRS * plane + i] = 1.0;
                }
                let decided = match self.boards_[board_of(row, col)]{
                    Status::Open => None,
                    Status::Won(m) if m == mine => Some(PLANE_WON_MINE),
                    Status::Won(_) => Some(PLANE_WON_THEIRS),
                    Status::Tied => Some(PLANE_TIED),
                };
                if let Some(p) = decided{
                    t[p * plane + i] = 1.0;
                }
            }
        }
        return t;
    }

    fn legal_actions_mask(&self) -> Vec<bool>{
        return (0..Self::ACTN_SPC_SIZE).map(|a| self.is_legal_action(a)).collect();
    }

    fn apply_action(&mut self, action: Action){
        trace!("Applying action {}", action);
        assert!(!self.is_terminal(), "apply_action({}) called on a finished game\n{}", action, self);
        assert!(self.is_legal_action(action), "Illegal action {} for board\n{}", action, self);
        let (row, col) = action_2_coord(action);
        let mark = self.active_mark();
        self.cells_[row][col] = mark;

        // Decide the small board, then the game.
        let b = board_of(row, col);
        self.boards_[b] = self.small_board_status(b);
        let won = |m: Mark| LINES.iter().any(|line| line.iter().all(|&i| self.boards_[i] == Status::Won(m)));
        self.status_ = if won(mark){
            Status::Won(mark)
        }else if self.boards_.iter().all(|&s| s != Status::Open){
            Status::Tied
        }else{
            Status::Open
        };

        // The opponent is sent to the board matching the cell just played.
        let target = (row % 3) * 3 + col % 3;
        self.next_ = (self.boards_[target] == Status::Open).then_some(target);
        self.x_to_move_ = !self.x_to_move_;
    }

    fn is_terminal(&self) -> bool{
        return self.status_ != Status::Open;
    }

    // Only the player who just moved can have won, so this is never +1.
    fn get_terminal_value(&self) -> f32{
        match self.status_{
            Status::Won(_) => -1.0,
            Status::Tied => 0.0,
            Status::Open => panic!("get_terminal_value called on a non-terminal state\n{}", self),
        }
    }
}

impl UtttState{
    pub fn new() -> Self{
        return Self{
            cells_: [[Mark::Empty; COLS]; ROWS],
            boards_: [Status::Open; 9],
            next_: None,
            x_to_move_: true,
            status_: Status::Open,
        };
    }

    // Current player marks (row, col). Returns false, changing nothing, if
    // that isn't a legal move.
    pub fn play(&mut self, row: usize, col: usize) -> bool{
        if row >= ROWS || col >= COLS{
            return false;
        }
        let action = coord_2_action(row, col);
        if !self.is_legal_action(action){
            return false;
        }
        self.apply_action(action);
        return true;
    }

    pub fn is_legal_action(&self, action: Action) -> bool{
        if self.is_terminal() || action >= ROWS * COLS{
            return false;
        }
        let (row, col) = action_2_coord(action);
        let b = board_of(row, col);
        return self.cells_[row][col] == Mark::Empty
            && self.boards_[b] == Status::Open
            && self.next_.is_none_or(|n| n == b);
    }

    pub fn x_to_move(&self) -> bool{
        return self.x_to_move_;
    }

    fn active_mark(&self) -> Mark{
        return if self.x_to_move_ {Mark::X} else {Mark::O};
    }

    fn inactive_mark(&self) -> Mark{
        return if self.x_to_move_ {Mark::O} else {Mark::X};
    }

    // Small board b's status, worked out from its cells.
    fn small_board_status(&self, b: usize) -> Status{
        let cell = |i: usize| self.cells_[(b / 3) * 3 + i / 3][(b % 3) * 3 + i % 3];
        for m in [Mark::X, Mark::O]{
            if LINES.iter().any(|line| line.iter().all(|&i| cell(i) == m)){
                return Status::Won(m);
            }
        }
        if (0..9).all(|i| cell(i) != Mark::Empty){
            return Status::Tied;
        }
        return Status::Open;
    }
}

// The small board (0-8, row-major) that cell (row, col) belongs to.
fn board_of(row: usize, col: usize) -> usize{
    return (row / 3) * 3 + col / 3;
}

pub fn coord_2_action(row: usize, col: usize) -> Action{
    return row * COLS + col;
}

pub fn action_2_coord(action: Action) -> (usize, usize){
    return (action / COLS, action % COLS);
}

fn mark_char(m: Mark) -> char{
    match m{
        Mark::Empty => '.',
        Mark::X => 'X',
        Mark::O => 'O',
    }
}


impl TextGame for UtttState{
    const PLAYERS: [&'static str; 2] = ["X", "O"];
    const MOVE_FORMAT: &'static str = "row, col, each 0-8";

    fn active_player(&self) -> &'static str{
        return if self.x_to_move_ {"X"} else {"O"};
    }

    fn parse_action(&self, input: &str) -> Result<Action, String>{
        let (row, col) = parse_coord(input)?;
        if row >= ROWS || col >= COLS{
            return Err(format!("({}, {}) is off the 9x9 board", row, col));
        }
        let action = coord_2_action(row, col);
        if self.is_legal_action(action){
            return Ok(action);
        }
        let b = board_of(row, col);
        return Err(if self.cells_[row][col] != Mark::Empty{
            format!("({}, {}) is already taken", row, col)
        }else if self.boards_[b] != Status::Open{
            format!("the small board at ({}, {}) is already decided", row, col)
        }else if let Some(n) = self.next_.filter(|&n| n != b){
            format!("you must play in the small board at rows {}-{}, cols {}-{}",
                    (n / 3) * 3, (n / 3) * 3 + 2, (n % 3) * 3, (n % 3) * 3 + 2)
        }else{
            format!("({}, {}) is not a legal move", row, col)
        });
    }

    fn action_text(&self, action: Action) -> String{
        let (row, col) = action_2_coord(action);
        return format!("{}, {}", row, col);
    }
}

impl WebGame for UtttState{
    // {"kind": "uttt", "board": one string per row ('.', 'X', 'O'),
    //  "boards": each small board, row-major: "." open, "X"/"O" won, "T" tied,
    //  "next": the small board the player to move must play in, or null}
    fn position_json(&self) -> serde_json::Value{
        let board: Vec<String> = self.cells_.iter().map(|row| row.iter().map(|&m| mark_char(m)).collect()).collect();
        let boards: Vec<String> = self.boards_.iter().map(|s| match s{
            Status::Open => ".".to_string(),
            Status::Won(m) => mark_char(*m).to_string(),
            Status::Tied => "T".to_string(),
        }).collect();
        return serde_json::json!({ "kind": "uttt", "board": board, "boards": boards, "next": self.next_ });
    }

    fn action_json(&self, action: Action) -> serde_json::Value{
        let (row, col) = action_2_coord(action);
        return serde_json::json!({ "row": row, "col": col });
    }
}


impl fmt::Display for UtttState{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result{
        writeln!(f, "    0 1 2   3 4 5   6 7 8")?;
        for row in 0..ROWS{
            if row > 0 && row % 3 == 0{
                writeln!(f, "   -------+-------+-------")?;
            }
            write!(f, "{}  ", row)?;
            for col in 0..COLS{
                if col > 0 && col % 3 == 0{
                    write!(f, " |")?;
                }
                write!(f, " {}", mark_char(self.cells_[row][col]))?;
            }
            writeln!(f)?;
        }
        let boards: Vec<String> = self.boards_.iter().map(|s| match s{
            Status::Open => ".".to_string(),
            Status::Won(m) => mark_char(*m).to_string(),
            Status::Tied => "T".to_string(),
        }).collect();
        writeln!(f, "Small boards: {} / {} / {}", boards[0..3].concat(), boards[3..6].concat(), boards[6..9].concat())?;
        match self.status_{
            Status::Open => match self.next_{
                Some(n) => writeln!(f, "{} to move, in the small board at rows {}-{}, cols {}-{}",
                                    self.active_player(), (n / 3) * 3, (n / 3) * 3 + 2, (n % 3) * 3, (n % 3) * 3 + 2),
                None => writeln!(f, "{} to move, in any open small board", self.active_player()),
            },
            Status::Won(m) => writeln!(f, "{} wins", mark_char(m)),
            Status::Tied => writeln!(f, "Draw"),
        }
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand::rngs::StdRng;

    fn legal_actions(s: &UtttState) -> Vec<Action>{
        return s.legal_actions_mask().iter().enumerate().filter(|(_, &l)| l).map(|(a, _)| a).collect();
    }

    // The cells of small board b.
    fn cells_of(b: usize) -> Vec<Action>{
        let mut v: Vec<Action> = (0..9).map(|i| coord_2_action((b / 3) * 3 + i / 3, (b % 3) * 3 + i % 3)).collect();
        v.sort();
        return v;
    }

    #[test]
    fn x_starts_and_may_play_anywhere(){
        let s = UtttState::new();
        assert_eq!(UtttState::NAME, "uttt");
        assert!(s.x_to_move());
        assert_eq!(s.active_player(), "X");
        assert_eq!(legal_actions(&s).len(), 81);
    }

    #[test]
    fn your_cell_picks_the_opponents_board(){
        let mut s = UtttState::new();
        assert!(s.play(1, 1)); // the middle cell of the top-left board
        assert_eq!(legal_actions(&s), cells_of(4), "O is sent to the center board");
        assert!(!s.play(0, 0), "outside the center board");
        assert!(s.play(3, 5)); // top-right cell of the center board
        assert_eq!(legal_actions(&s), cells_of(2), "X is sent to the top-right board");
    }

    #[test]
    fn being_sent_to_your_own_cell_is_allowed(){
        // Playing the center cell of the center board sends the opponent back there.
        let mut s = UtttState::new();
        assert!(s.play(4, 4));
        let legal = legal_actions(&s);
        assert_eq!(legal.len(), 8);
        assert!(legal.iter().all(|a| cells_of(4).contains(a)));
    }

    #[test]
    fn winning_a_small_board_closes_it_and_frees_whoever_is_sent_there(){
        let mut s = UtttState::new();
        // X completes the top row of the top-left board.
        s.cells_[0][0] = Mark::X;
        s.cells_[0][1] = Mark::X;
        s.next_ = Some(0);
        assert!(s.play(0, 2));
        assert_eq!(s.boards_[0], Status::Won(Mark::X));
        assert!(!s.is_terminal());
        // (0, 2) is a top-right cell, so O is sent to the top-right board.
        assert_eq!(legal_actions(&s), cells_of(2));
        assert!(s.play(1, 6)); // O sends X to board 3 (middle-left)
        assert!(s.play(4, 0)); // X sends O to board 3 too
        // O's top-left cell would send X to the won top-left board, so X may go anywhere open.
        assert!(s.play(3, 0));
        let legal = legal_actions(&s);
        assert!(legal.iter().all(|&a| board_of(a / 9, a % 9) != 0), "no moves in a decided board");
        assert_eq!(legal.len(), 72 - 3, "every empty cell outside the top-left board");
    }

    #[test]
    fn a_full_small_board_without_a_line_is_tied(){
        let mut s = UtttState::new();
        // Board 4 (center) filled X O X / X O O / O X . with X to move there.
        let pattern = [Mark::X, Mark::O, Mark::X, Mark::X, Mark::O, Mark::O, Mark::O, Mark::X];
        for (i, &m) in pattern.iter().enumerate(){
            s.cells_[3 + i / 3][3 + i % 3] = m;
        }
        s.next_ = Some(4);
        assert!(s.play(5, 5));
        assert_eq!(s.boards_[4], Status::Tied);
        assert!(!s.is_terminal());
    }

    #[test]
    fn three_small_boards_in_a_row_wins(){
        let mut s = UtttState::new();
        s.boards_[0] = Status::Won(Mark::X);
        s.boards_[4] = Status::Won(Mark::X);
        // X completes the bottom-right board's diagonal, winning it and the game.
        s.cells_[6][6] = Mark::X;
        s.cells_[7][7] = Mark::X;
        s.next_ = Some(8);
        assert!(s.play(8, 8));
        assert_eq!(s.status_, Status::Won(Mark::X));
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), -1.0, "O, to move, has lost");
        assert!(legal_actions(&s).is_empty());
    }

    #[test]
    fn deciding_every_board_without_a_line_is_a_draw(){
        let mut s = UtttState::new();
        // Small boards X O X / O X X / O X _, and the last one gets tied.
        let won = [Mark::X, Mark::O, Mark::X, Mark::O, Mark::X, Mark::X, Mark::O, Mark::X];
        for (b, &m) in won.iter().enumerate(){
            s.boards_[b] = Status::Won(m);
        }
        let pattern = [Mark::X, Mark::O, Mark::X, Mark::X, Mark::O, Mark::O, Mark::O, Mark::X];
        for (i, &m) in pattern.iter().enumerate(){
            s.cells_[6 + i / 3][6 + i % 3] = m;
        }
        s.next_ = Some(8);
        assert!(s.play(8, 8));
        assert_eq!(s.boards_[8], Status::Tied);
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), 0.0);
    }

    #[test]
    fn parse_action_says_why_a_move_is_illegal(){
        let mut s = UtttState::new();
        assert_eq!(s.parse_action("1, 1"), Ok(coord_2_action(1, 1)));
        s.play(1, 1);
        assert!(s.parse_action("1, 1").unwrap_err().contains("taken"));
        assert!(s.parse_action("0, 0").unwrap_err().contains("rows 3-5, cols 3-5"));
        assert!(s.parse_action("9, 0").unwrap_err().contains("off"));
        assert_eq!(s.parse_action(&s.action_text(coord_2_action(4, 4))), Ok(coord_2_action(4, 4)));
    }

    #[test]
    fn to_tensor_is_from_the_side_to_moves_perspective(){
        let mut s = UtttState::new();
        s.play(1, 1); // X; O to move, sent to the center board
        let t = s.to_tensor();
        let plane = 81;
        assert_eq!(t.len(), UtttState::N_CHAN * plane);
        let at = |p: usize, r: usize, c: usize| t[p * plane + r * 9 + c];
        assert_eq!(at(PLANE_THEIRS, 1, 1), 1.0, "X's mark, seen by O");
        assert_eq!(at(PLANE_MINE, 1, 1), 0.0);
        // The forced board isn't encoded: nothing marks the center board O must play in.
        assert_eq!(t.iter().filter(|&&x| x != 0.0).count(), 1, "only X's one mark");
    }

    #[test]
    fn random_playouts_keep_boards_consistent(){
        let mut rng = StdRng::seed_from_u64(3);
        let (mut x, mut o, mut draws) = (0, 0, 0);
        for _ in 0..300{
            let mut s = UtttState::new();
            let mut plies = 0;
            while !s.is_terminal(){
                let legal = legal_actions(&s);
                assert!(!legal.is_empty(), "no legal move\n{s}");
                if let Some(n) = s.next_{
                    assert!(legal.iter().all(|&a| board_of(a / 9, a % 9) == n), "left the forced board\n{s}");
                }
                s.apply_action(legal[rng.random_range(0..legal.len())]);
                plies += 1;
                assert!(plies <= 81);
                for b in 0..9{
                    assert_eq!(s.boards_[b], s.small_board_status(b), "board {b}\n{s}");
                }
            }
            match s.status_{
                Status::Won(Mark::X) => x += 1,
                Status::Won(_) => o += 1,
                _ => draws += 1,
            }
        }
        assert!(x > 0 && o > 0 && draws > 0, "X {x}, O {o}, draws {draws}");
    }
}
