//! Pente, the official game: a 19x19 board, White moves first.
//!
//! Players alternate placing a stone on an empty intersection. A stone that
//! flanks exactly two enemy stones in a line (X O O X, in any of the 8
//! directions) captures that pair; moving into a flanked position yourself
//! is safe. A player wins by making five or more in a row (horizontally,
//! vertically or diagonally) or by capturing five pairs. A full board with
//! no winner is a draw.
//!
//! Tournament rule: White's first stone must go on the center point, and
//! White's second stone must be at least three intersections away from it,
//! i.e. outside the central 5x5 square.

use std::fmt;
use tracing::trace;

use crate::games::{parse_coord, Action, GameState, TextGame, WebGame};
use crate::model::NetConfig;

pub const ROWS: usize = 19;
pub const COLS: usize = 19;
const CENTER: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    Empty,
    Black,
    White,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Ongoing,
    Won(Piece), // Winner's color
    Draw,
}

#[derive(Clone, Debug)]
pub struct PenteState {
    board_: [[Piece; COLS]; ROWS],
    b_caps_: u8,        // pairs captured by Black
    w_caps_: u8,        // pairs captured by White
    is_black_: bool,    // Black to move
    num_pieces_: usize, // stones currently on the board
    num_moves_: usize,  // stones placed so far (plies played), for the tournament rule
    status_: Status,
}

// Default network size for a fresh Pente model. Not tuned. 10 residual
// blocks stack 21 3x3 convs, enough for every point to see the whole
// 19x19 board (6 blocks would not be).
pub fn default_net_config() -> NetConfig {
    return NetConfig {
        channels: 128,
        blocks: 10,
        value_hidden: 128,
    };
}

impl Default for PenteState {
    fn default() -> Self {
        return Self::new();
    }
}

impl GameState for PenteState {
    const NAME: &'static str = "pente";
    const ACTN_SPC_SIZE: usize = ROWS * COLS;
    const N_CHAN: usize = 4;
    const ROWS: usize = ROWS;
    const COLS: usize = COLS;

    // Output (n_chan) x ROWS x COLS Tensor.
    // channel 0 is occupancy mask of current player
    // channel 1 is captures of current player
    // channel 2 is occupancy mask of opponent
    // channel 3 is captures of opponent
    fn to_tensor(&self) -> Vec<f32> {
        let plane = ROWS * COLS;
        let mut a = vec![0.0; Self::N_CHAN * plane];
        let (b_ofst, w_ofst) = if self.is_black_ { (0, 2) } else { (2, 0) };
        for r in 0..ROWS {
            for c in 0..COLS {
                match self.board_[r][c] {
                    Piece::Black => a[b_ofst * plane + r * COLS + c] = 1.0,
                    Piece::White => a[w_ofst * plane + r * COLS + c] = 1.0,
                    Piece::Empty => {}
                }
            }
        }
        a[(b_ofst + 1) * plane..(b_ofst + 2) * plane].fill(self.b_caps_ as f32);
        a[(w_ofst + 1) * plane..(w_ofst + 2) * plane].fill(self.w_caps_ as f32);
        return a;
    }

    fn legal_actions_mask(&self) -> Vec<bool> {
        return (0..Self::ACTN_SPC_SIZE)
            .map(|i| self.is_legal_action(i))
            .collect();
    }

    fn apply_action(&mut self, action: Action) {
        trace!("Applying action {}", action);
        assert!(
            !self.is_terminal(),
            "apply_action({}) called on a finished game\n{}",
            action,
            self
        );
        assert!(
            self.is_legal_action(action),
            "Illegal action {} for board\n{}",
            action,
            self
        );
        let (row, col) = action_2_coord(action);
        self.board_[row][col] = self.active_piece();
        self.num_pieces_ += 1;
        self.num_moves_ += 1;
        self.update_captures(row, col);
        if self.b_caps_ >= 5 || self.w_caps_ >= 5 || self.is_con5(row, col) {
            self.status_ = Status::Won(self.active_piece());
        } else if self.num_pieces_ == ROWS * COLS {
            self.status_ = Status::Draw;
        }
        self.is_black_ = !self.is_black_;
    }

    fn is_terminal(&self) -> bool {
        return self.status_ != Status::Ongoing;
    }

    // In Pente the player who just moved is the one who can win, so this is
    // never +1: it's -1 (side to move lost) or 0 (draw).
    fn get_terminal_value(&self) -> f32 {
        return match self.status_ {
            Status::Won(_) => -1.0,
            Status::Draw => 0.0,
            Status::Ongoing => panic!(
                "get_terminal_value called on a non-terminal state\n{}",
                self
            ),
        };
    }
}

impl PenteState {
    pub fn new() -> Self {
        trace!("Pente new()");
        return Self {
            board_: [[Piece::Empty; COLS]; ROWS],
            b_caps_: 0,
            w_caps_: 0,
            is_black_: false, // White moves first
            num_pieces_: 0,
            num_moves_: 0,
            status_: Status::Ongoing,
        };
    }

    // Current player plays at (row, col). Returns false, changing nothing,
    // if that isn't a legal move.
    pub fn play(&mut self, row: usize, col: usize) -> bool {
        // Bounds-check before flattening: e.g. (0, COLS) would otherwise
        // wrap to the action for (1, 0).
        if row >= ROWS || col >= COLS {
            return false;
        }
        let action = coord_2_action(row, col);
        if self.is_legal_action(action) {
            self.apply_action(action);
            return true;
        }
        return false;
    }

    // No action is legal once the game is over.
    pub fn is_legal_action(&self, action: Action) -> bool {
        if self.is_terminal() || action >= ROWS * COLS {
            return false;
        }
        let (row, col) = action_2_coord(action);
        if self.board_[row][col] != Piece::Empty {
            return false;
        }
        // Tournament rule, for White's first two stones.
        return match self.num_moves_ {
            0 => row == CENTER && col == CENTER,
            2 => row.abs_diff(CENTER) >= 3 || col.abs_diff(CENTER) >= 3,
            _ => true,
        };
    }

    // Is current player black
    pub fn is_black(&self) -> bool {
        return self.is_black_;
    }

    // Would the current player's stone at (row, col) make five or more in a row?
    fn is_con5(&self, row: usize, col: usize) -> bool {
        let active = self.active_piece();
        let (r, c) = (row as i32, col as i32);
        const DIRS: [(i32, i32); 4] = [
            (1, 0),
            (0, 1), // Vert/Horiz
            (1, 1),
            (1, -1), // Diagonals
        ];
        for (dr, dc) in DIRS {
            let mut count = 1;
            for sign in [1, -1] {
                for i in 1..5 {
                    if self.get_square(r + sign * dr * i, c + sign * dc * i) == Some(active) {
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            if count >= 5 {
                return true;
            }
        }
        return false;
    }

    // The piece at (row, col), or None if that's off the board.
    fn get_square(&self, row: i32, col: i32) -> Option<Piece> {
        if row < 0 || row >= ROWS as i32 || col < 0 || col >= COLS as i32 {
            return None;
        }
        return Some(self.board_[row as usize][col as usize]);
    }

    // Removes every pair the current player's stone at (row, col) captures.
    fn update_captures(&mut self, row: usize, col: usize) {
        let active = self.active_piece();
        let inactive = self.inactive_piece();
        let (r, c) = (row as i32, col as i32);
        const DIRS: [(i32, i32); 8] = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1), // Vert/Horiz
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1), // Diagonals
        ];
        for (dr, dc) in DIRS {
            if self.get_square(r + dr, c + dc) == Some(inactive)
                && self.get_square(r + 2 * dr, c + 2 * dc) == Some(inactive)
                && self.get_square(r + 3 * dr, c + 3 * dc) == Some(active)
            {
                self.board_[(r + dr) as usize][(c + dc) as usize] = Piece::Empty;
                self.board_[(r + 2 * dr) as usize][(c + 2 * dc) as usize] = Piece::Empty;
                if self.is_black_ {
                    self.b_caps_ += 1;
                } else {
                    self.w_caps_ += 1;
                }
                self.num_pieces_ -= 2;
            }
        }
    }

    fn active_piece(&self) -> Piece {
        return if self.is_black_ {
            Piece::Black
        } else {
            Piece::White
        };
    }

    fn inactive_piece(&self) -> Piece {
        return if self.is_black_ {
            Piece::White
        } else {
            Piece::Black
        };
    }
}

pub fn coord_2_action(row: usize, col: usize) -> Action {
    return row * COLS + col;
}

pub fn action_2_coord(action: Action) -> (usize, usize) {
    return (action / COLS, action % COLS);
}

impl TextGame for PenteState {
    const PLAYERS: [&'static str; 2] = ["White", "Black"];
    const MOVE_FORMAT: &'static str = "row, col";

    fn active_player(&self) -> &'static str {
        return if self.is_black_ { "Black" } else { "White" };
    }

    fn parse_action(&self, input: &str) -> Result<Action, String> {
        let (row, col) = parse_coord(input)?;
        if row >= ROWS || col >= COLS {
            return Err(format!(
                "({}, {}) is off the {}x{} board",
                row, col, ROWS, COLS
            ));
        }
        let action = coord_2_action(row, col);
        if !self.is_legal_action(action) {
            return Err(format!("({}, {}) is not a legal move", row, col));
        }
        return Ok(action);
    }

    fn action_text(&self, action: Action) -> String {
        let (row, col) = action_2_coord(action);
        return format!("{}, {}", row, col);
    }
}

impl WebGame for PenteState {
    // {"kind": "grid", "rows", "cols", "board": one string per row ('.', 'B', 'W'),
    //  "captures": pairs captured by each player}
    fn position_json(&self) -> serde_json::Value {
        let board: Vec<String> = (0..ROWS)
            .map(|r| {
                (0..COLS)
                    .map(|c| match self.board_[r][c] {
                        Piece::Empty => '.',
                        Piece::Black => 'B',
                        Piece::White => 'W',
                    })
                    .collect()
            })
            .collect();
        return serde_json::json!({
            "kind": "grid",
            "rows": ROWS,
            "cols": COLS,
            "board": board,
            "captures": { "Black": self.b_caps_, "White": self.w_caps_ },
        });
    }

    fn action_json(&self, action: Action) -> serde_json::Value {
        let (row, col) = action_2_coord(action);
        return serde_json::json!({ "row": row, "col": col });
    }
}

impl fmt::Display for PenteState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "   ")?;
        for col in 0..COLS {
            write!(f, "{:02} ", col)?;
        }
        writeln!(f)?;
        for row in 0..ROWS {
            write!(f, "{:02}  ", row)?;
            for col in 0..COLS {
                match self.board_[row][col] {
                    Piece::Empty => write!(f, "·  ")?,
                    Piece::Black => write!(f, "B  ")?,
                    Piece::White => write!(f, "W  ")?,
                }
            }
            writeln!(f)?;
        }
        writeln!(
            f,
            "White captures:{}   Black captures:{}",
            self.w_caps_, self.b_caps_
        )?;
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    // Builds a position from rows of '.', 'B', 'W' (missing rows/cols are
    // empty). The move count is set to the number of stones, so positions
    // with 3+ stones are past the tournament rule.
    fn board(rows: &[&str], black_to_move: bool) -> PenteState {
        let mut s = PenteState::new();
        for (r, row) in rows.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                s.board_[r][c] = match ch {
                    'B' => Piece::Black,
                    'W' => Piece::White,
                    _ => Piece::Empty,
                };
                if s.board_[r][c] != Piece::Empty {
                    s.num_pieces_ += 1;
                }
            }
        }
        s.num_moves_ = s.num_pieces_;
        s.is_black_ = black_to_move;
        return s;
    }

    // `n` empty rows, then `row`: for positions along the bottom edge.
    fn after_rows(n: usize, row: &str) -> Vec<String> {
        let mut v = vec![String::new(); n];
        v.push(row.to_string());
        return v;
    }

    fn strs(v: &[String]) -> Vec<&str> {
        return v.iter().map(|s| s.as_str()).collect();
    }

    fn legal_actions(s: &PenteState) -> Vec<Action> {
        return s
            .legal_actions_mask()
            .iter()
            .enumerate()
            .filter(|(_, &legal)| legal)
            .map(|(a, _)| a)
            .collect();
    }

    #[test]
    fn new_game_is_19x19_with_white_to_move() {
        let s = PenteState::new();
        assert_eq!(PenteState::NAME, "pente");
        assert_eq!(
            (
                PenteState::ROWS,
                PenteState::COLS,
                PenteState::ACTN_SPC_SIZE
            ),
            (19, 19, 361)
        );
        assert!(!s.is_black());
        assert_eq!(s.active_player(), "White");
        assert!(!s.is_terminal());
        assert_eq!(s.legal_actions_mask().len(), 361);
    }

    #[test]
    fn first_move_must_be_the_center() {
        let mut s = PenteState::new();
        assert_eq!(legal_actions(&s), vec![coord_2_action(9, 9)]);
        assert!(!s.play(0, 0));
        assert!(!s.play(9, 10));
        assert!(
            !s.is_black(),
            "a rejected move must not change the side to move"
        );
        assert!(s.play(9, 9));
        assert_eq!(s.board_[9][9], Piece::White);
        assert!(s.is_black());
    }

    #[test]
    fn blacks_first_move_is_unrestricted() {
        let mut s = PenteState::new();
        assert!(s.play(9, 9));
        assert_eq!(legal_actions(&s).len(), 360);
        assert!(!s.play(9, 9), "occupied square must be rejected");
        assert!(s.play(9, 10), "right next to the center is fine for Black");
        assert_eq!(s.board_[9][10], Piece::Black);
    }

    #[test]
    fn whites_second_move_is_at_least_three_from_center() {
        let mut s = PenteState::new();
        assert!(s.play(9, 9));
        assert!(s.play(9, 10));
        let mask = s.legal_actions_mask();
        for r in 0..ROWS {
            for c in 0..COLS {
                let outside_5x5 = r.abs_diff(9) >= 3 || c.abs_diff(9) >= 3;
                assert_eq!(mask[coord_2_action(r, c)], outside_5x5, "({r}, {c})");
            }
        }
        assert!(!s.play(7, 7), "corner of the central 5x5");
        assert!(!s.play(11, 9));
        assert!(s.play(6, 9), "three rows from the center");
        // From the fourth stone on, any empty point is legal again.
        assert_eq!(legal_actions(&s).len(), 361 - 3);
        assert!(s.play(8, 8));
        assert!(s.play(10, 10));
    }

    #[test]
    fn play_rejects_out_of_bounds_without_wrapping() {
        let mut s = board(
            &[
                "W", "", "", "", "", "", "", "", "", "", "", "", "B", "", "", "W",
            ],
            false,
        );
        // (0, COLS) would flatten to the action for (1, 0).
        assert!(!s.play(0, COLS));
        assert!(!s.play(ROWS, 0));
        assert_eq!(s.board_[1][0], Piece::Empty);
        assert!(!s.is_legal_action(ROWS * COLS));
        assert!(s.play(1, 0));
    }

    #[test]
    fn action_coord_roundtrip() {
        for a in 0..ROWS * COLS {
            let (r, c) = action_2_coord(a);
            assert!(r < ROWS && c < COLS);
            assert_eq!(coord_2_action(r, c), a);
        }
    }

    // White completes five by playing (r, c).
    fn assert_white_wins_by_playing(rows: &[&str], r: usize, c: usize) {
        let mut s = board(rows, false);
        assert!(s.play(r, c));
        assert!(
            s.is_terminal(),
            "expected a win after playing ({r}, {c})\n{s}"
        );
        assert_eq!(
            s.get_terminal_value(),
            -1.0,
            "side to move after a win has lost"
        );
        assert!(s.legal_actions_mask().iter().all(|&l| !l));
    }

    #[test]
    fn five_in_a_row_wins_in_every_direction() {
        // horizontal, filling the middle of the line
        assert_white_wins_by_playing(&["", "", "..WW.WW.."], 2, 4);
        // vertical, filling the end
        assert_white_wins_by_playing(&[".W", ".W", ".W", ".W"], 4, 1);
        // diagonal
        assert_white_wins_by_playing(&["W", ".W", "..W", "...W"], 4, 4);
        // anti-diagonal
        assert_white_wins_by_playing(&["....W", "...W", "..W", ".W"], 4, 0);
        // along the far edge
        let edge = after_rows(18, &format!("{}WWWW", ".".repeat(14))); // (18, 14..=17)
        assert_white_wins_by_playing(&strs(&edge), 18, 18);
    }

    #[test]
    fn overline_of_six_also_wins() {
        assert_white_wins_by_playing(&["WWW.WW"], 0, 3);
    }

    #[test]
    fn four_in_a_row_or_a_broken_line_does_not_win() {
        let mut s = board(&["WWW"], false);
        assert!(s.play(0, 3));
        assert!(!s.is_terminal());
        // An opponent stone breaks the line.
        let mut s = board(&["WWBW."], false);
        assert!(s.play(0, 4));
        assert!(!s.is_terminal());
    }

    #[test]
    fn captures_a_flanked_pair_in_every_direction() {
        const DIRS: [(i32, i32); 8] = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
        ];
        for (dr, dc) in DIRS {
            // White plays at (9, 9); B B then W lie along the direction.
            let at = |i: i32| ((9 + i * dr) as usize, (9 + i * dc) as usize);
            let mut s = PenteState::new();
            s.board_[at(1).0][at(1).1] = Piece::Black;
            s.board_[at(2).0][at(2).1] = Piece::Black;
            s.board_[at(3).0][at(3).1] = Piece::White;
            s.num_pieces_ = 3;
            s.num_moves_ = 4; // past the tournament rule
            assert!(s.play(9, 9));
            assert_eq!(s.w_caps_, 1, "direction ({dr}, {dc})");
            assert_eq!(s.board_[at(1).0][at(1).1], Piece::Empty);
            assert_eq!(s.board_[at(2).0][at(2).1], Piece::Empty);
            assert_eq!(s.num_pieces_, 2, "two stones removed, one added");
        }
    }

    #[test]
    fn captures_reach_the_far_corner() {
        let rows = after_rows(18, &format!("{}BWW", ".".repeat(15))); // (18, 15..=17)
        let mut s = board(&strs(&rows), true);
        assert!(s.play(18, 18));
        assert_eq!(s.b_caps_, 1);
        assert_eq!(s.board_[18][16], Piece::Empty);
        assert_eq!(s.board_[18][17], Piece::Empty);
    }

    #[test]
    fn one_move_can_capture_two_pairs() {
        let mut s = board(&["WBB.BBW"], false);
        assert!(s.play(0, 3));
        assert_eq!(s.w_caps_, 2);
        assert_eq!(s.num_pieces_, 3);
    }

    #[test]
    fn three_stones_or_unflanked_pairs_are_not_captured() {
        let mut s = board(&["WBBB.", "", "W"], false);
        assert!(s.play(0, 4));
        assert_eq!(s.w_caps_, 0);
        let mut s = board(&[".BB.", "", "W"], false);
        assert!(s.play(0, 3));
        assert_eq!(s.w_caps_, 0);
    }

    #[test]
    fn moving_into_a_flanked_position_is_safe() {
        // Black completes W B B W itself: no capture.
        let mut s = board(&["WB.W"], true);
        assert!(s.play(0, 2));
        assert_eq!(s.b_caps_ + s.w_caps_, 0);
        assert_eq!(s.board_[0][1], Piece::Black);
        assert_eq!(s.board_[0][2], Piece::Black);
    }

    #[test]
    fn fifth_capture_wins() {
        // Black already has 4 captures and captures a fifth pair.
        let mut s = board(&["BWW.", "", "W"], true);
        s.b_caps_ = 4;
        assert!(s.play(0, 3));
        assert_eq!(s.b_caps_, 5);
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), -1.0);
        // A fourth capture is not enough.
        let mut s = board(&["BWW.", "", "W"], true);
        s.b_caps_ = 3;
        assert!(s.play(0, 3));
        assert_eq!(s.b_caps_, 4);
        assert!(!s.is_terminal());
    }

    #[test]
    fn full_board_without_a_win_is_a_draw() {
        // Every line through this pattern reads ...BBBBW BBBBW...: no five,
        // and no pair to capture. (18, 18) is left for White to fill.
        let mut s = PenteState::new();
        for r in 0..ROWS {
            for c in 0..COLS {
                s.board_[r][c] = if (c + 2 * r) % 5 == 4 {
                    Piece::White
                } else {
                    Piece::Black
                };
            }
        }
        assert_eq!(s.board_[18][18], Piece::White);
        s.board_[18][18] = Piece::Empty;
        s.num_pieces_ = ROWS * COLS - 1;
        s.num_moves_ = s.num_pieces_;
        s.is_black_ = false;
        assert!(s.play(18, 18));
        assert_eq!(s.b_caps_ + s.w_caps_, 0);
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), 0.0);
    }

    #[test]
    fn finished_game_rejects_further_moves() {
        let mut s = board(&["WWWW"], false);
        assert!(s.play(0, 4));
        assert!(!s.play(5, 5));
        assert!(!s.is_legal_action(coord_2_action(5, 5)));
    }

    #[test]
    #[should_panic(expected = "finished game")]
    fn apply_action_panics_after_game_over() {
        let mut s = board(&["WWWW"], false);
        s.apply_action(coord_2_action(0, 4));
        s.apply_action(coord_2_action(5, 5));
    }

    #[test]
    #[should_panic(expected = "Illegal action")]
    fn apply_action_panics_on_an_illegal_move() {
        // Off-center first move, against the tournament rule.
        PenteState::new().apply_action(0);
    }

    #[test]
    #[should_panic(expected = "non-terminal")]
    fn terminal_value_panics_on_ongoing_game() {
        PenteState::new().get_terminal_value();
    }

    #[test]
    fn to_tensor_is_from_the_side_to_moves_perspective() {
        let plane = ROWS * COLS;
        let at = |chan: usize, r: usize, c: usize| chan * plane + r * COLS + c;
        let mut s = PenteState::new();
        s.play(9, 9); // White stone; Black to move
        let t = s.to_tensor();
        assert_eq!(t.len(), PenteState::N_CHAN * plane);
        assert_eq!(t[at(0, 9, 9)], 0.0, "channel 0 is the side to move (Black)");
        assert_eq!(t[at(2, 9, 9)], 1.0, "channel 2 is the opponent (White)");
        // Captures fill the whole plane of the capturing player's channel.
        let mut s = board(&["WBB.", "", "B"], false);
        s.play(0, 3); // White captures; Black to move
        let t = s.to_tensor();
        assert!(
            t[at(3, 0, 0)..at(4, 0, 0)].iter().all(|&x| x == 1.0),
            "opponent captures"
        );
        assert!(
            t[at(1, 0, 0)..at(2, 0, 0)].iter().all(|&x| x == 0.0),
            "own captures"
        );
    }

    #[test]
    fn random_playouts_keep_the_board_consistent() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..30 {
            let mut s = PenteState::new();
            let mut plies = 0;
            while !s.is_terminal() {
                let legal = legal_actions(&s);
                assert!(
                    !legal.is_empty(),
                    "non-terminal position with no legal move\n{s}"
                );
                s.apply_action(legal[rng.random_range(0..legal.len())]);
                plies += 1;
                let stones = s
                    .board_
                    .iter()
                    .flatten()
                    .filter(|&&p| p != Piece::Empty)
                    .count();
                assert_eq!(s.num_pieces_, stones);
                assert_eq!(s.num_moves_, plies);
                // Each capture removes two stones, and at most 9 captures
                // happen before one side reaches 5.
                assert!(plies <= ROWS * COLS + 18, "game ran too long\n{s}");
            }
            assert!(legal_actions(&s).is_empty());
            let v = s.get_terminal_value();
            assert!(v == -1.0 || v == 0.0);
        }
    }
}
