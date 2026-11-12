use std::{fmt};
use tracing::trace;

use crate::gamestate::GameState;
use crate::tard::NetConfig;

pub const ROWS: usize = 9;
pub const COLS: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece{
    Empty = 0,
    Black = 1,
    White = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status{
    Ongoing,
    Won(Piece), // Winner's color
    Draw,
}

#[derive(Clone, Debug)]
pub struct PenteState{
    board_: [[Piece; COLS]; ROWS],
    b_caps_: u8, 
    w_caps_: u8,
    is_black_: bool,
    num_pieces_: usize, // Number of pieces actively on the board.
                        // Is redundant, since we could loop over board,
                        // but maintined to minimize computation
    status_: Status,
}

/// Default network size for a fresh Pente model. Chosen somewhat
/// arbitrarily, not tuned: it's smaller than the old 128/10/128 because that
/// seemed oversized for a 9x9 board.
pub fn default_net_config() -> NetConfig{
    return NetConfig{ channels: 64, blocks: 6, value_hidden: 64 };
}

impl Default for PenteState{
    fn default() -> Self {
        return Self::new();
    }
}

impl GameState for PenteState{
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
        let mut a = vec![0.0; PenteState::COLS * PenteState::ROWS * PenteState::N_CHAN];
        fn index(chan: usize, row: usize, col: usize) -> usize{
            return chan*PenteState::ROWS * PenteState::COLS +
                   row * PenteState::COLS + col;
        }
        let b_ofst = if self.is_black_ {0} else {2};
        let w_ofst = if self.is_black_ {2} else {0};
        for r in 0..PenteState::ROWS{
            for c in 0..PenteState::COLS{
                if self.board_[r][c] == Piece::Black{
                    a[index(b_ofst, r, c)] = 1.0;
                }else if self.board_[r][c] == Piece::White{
                    a[index(w_ofst, r, c)] = 1.0;
                }
                a[index(b_ofst + 1,r,c)] = self.b_caps_ as f32;
                a[index(w_ofst + 1,r,c)] = self.w_caps_ as f32;
            }
        }
        return a;
    }

    fn legal_actions_mask(&self) -> Vec<bool> {
        return (0..Self::ACTN_SPC_SIZE).map(|i| self.is_legal_action(i)).collect();
    }
    
    fn apply_action(&mut self, action: usize){
        trace!("Applying action {}", action);
        assert!(!self.is_terminal(), "apply_action({}) called on a finished game\n{}", action, self);
        assert!(self.is_legal_action(action), "Illegal action {} for board \n{}", action, self);
        let (row,col) = action_2_coord(action);
        if self.is_black_{
            self.board_[row][col] = Piece::Black;
        }else{
            self.board_[row][col] = Piece::White;
        }
        self.num_pieces_ += 1;
        self.update_captures(row, col);
        if self.b_caps_ >= 5 || self.w_caps_ >= 5 || self.is_con5(row,col){
            self.status_ = Status::Won(self.get_active_piece());
        }else if self.num_pieces_ == PenteState::ROWS * PenteState::COLS{
            self.status_ = Status::Draw;
        }
        self.is_black_ = !self.is_black_;
    }
    
    fn is_terminal(&self) -> bool{
        return self.status_ != Status::Ongoing;
    }
    

    // In Pente the player who just moved is the one who can win, so this is
    // never +1: it's -1 (side to move lost) or 0 (draw).
    fn get_terminal_value(&self) -> f32{
        match self.status_{
            Status::Won(_) => -1.0,
            Status::Draw => 0.0,
            Status::Ongoing => panic!("get_terminal_value called on a non-terminal state\n{}", self),
        }
    }
}


impl PenteState{
    pub fn new() -> Self{
        trace!("Pente new()");
        Self{
            board_: [[Piece::Empty; PenteState::COLS]; PenteState::ROWS],
            b_caps_:  0,
            w_caps_:  0,
            is_black_: true,
            num_pieces_: 0,
            status_: Status::Ongoing,
        }
    }

    // Current player plays at provided space
    pub fn play(&mut self, row: usize, col: usize) -> bool{
        // Bounds-check before flattening: e.g. (0, COLS) would otherwise
        // wrap to the legal action for (1, 0).
        if row >= PenteState::ROWS || col >= PenteState::COLS{
            return false;
        }
        let action = coord_2_action(row,col);
        if self.is_legal_action(action){
            self.apply_action(action);
            return true;
        }
        return false;
    }

    // No action is legal once the game is over.
    pub fn is_legal_action(&self, action: usize) -> bool{
        let (row, col) = action_2_coord(action);
        return !self.is_terminal() && row < PenteState::ROWS && col < PenteState::COLS && self.board_[row][col] == Piece::Empty;
    }
    
    pub fn active_plyr_str(&self) -> &str{
        const PLYR_MAP: [&str; 2] = ["White", "Black"];
        return PLYR_MAP[self.is_black_ as usize];
    }

    // Is current player black
    pub fn is_black(&self) -> bool{
        return self.is_black_;
    }

    // If current player played at row,col, would it be connect-5
    fn is_con5(&self, row: usize, col: usize) -> bool{
        let active = self.get_active_piece();
        let mut count:u32;
        let r = row as i32;
        let c = col as i32;
        const DIRS: [(i32, i32); 4] = [
            (1, 0), (0, 1),   // Vert/Horiz
            (1, 1), (1, -1),  // Diagonals
        ];
        
        for (dr, dc) in DIRS.iter(){
            count = 1;
            for i in 1..5{
                if self.get_square(r + dr * i, c + dc * i) == Some(active){
                    count+=1;
                } else{ break;}
            }
            for i in 1..5{
                if self.get_square(r - dr * i, c - dc * i) == Some(active){
                    count+=1;
                } else{ break;}
            }
            if count >= 5{
                return true;
            }
        }
        return false;
    }

    // Safe method to grab piece occupying space. 
    // If space is out of bounds it returns None.
    fn get_square(&self, row: i32, col: i32) -> Option<Piece>{
        if row < 0 || row >= (PenteState::ROWS as i32) || col < 0 || col >= (PenteState::COLS as i32) {
            return None
        }
        let r = row as usize;
        let c = col as usize;
        return Some(self.board_[r][c]);
    }

    // Check for captures due to current player playing at provided space,
    // and remove the captured pieces.
    fn update_captures(&mut self, row: usize, col: usize) {
        let active = self.get_active_piece();
        let inactive = self.get_inactive_piece();
        let r = row as i32;
        let c = col as i32;
        const DIRS: [(i32, i32); 8] = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   // Vert/Horiz
            (-1, -1), (1, 1), (-1, 1), (1, -1)  // Diagonals
        ];
        for (dr, dc) in DIRS.iter() {
            if self.get_square(r + dr, c + dc) == Some(inactive) &&
            self.get_square(r + dr * 2, c + dc * 2) == Some(inactive) &&
            self.get_square(r + dr * 3, c + dc * 3) == Some(active) 
            {
                self.board_[(r + dr) as usize][(c + dc) as usize] = Piece::Empty;
                self.board_[(r + dr * 2) as usize][(c + dc * 2) as usize] = Piece::Empty;
                self.add_captures(self.is_black_, 1);
                self.num_pieces_ -= 2;
            }
        }
    }
    
    // Change players captures by delta.
    fn add_captures(&mut self, is_black:bool, delta: u8){
        if is_black{
            self.b_caps_ += delta;
        }else{
            self.w_caps_ += delta;
        }
    }

    // Get active player as a Piece
    fn get_active_piece(&self) -> Piece{
        const PIECE_MAP: [Piece; 2] = [Piece::White, Piece::Black];
        return PIECE_MAP[self.is_black_ as usize]
    }

    // Get inactive player as a Piece
    fn get_inactive_piece(&self) -> Piece{
        const PIECE_MAP: [Piece; 2] = [Piece::Black, Piece::White];
        return PIECE_MAP[self.is_black_ as usize]
    }
}

pub fn coord_2_action(row: usize, col:usize) -> usize{
    row * PenteState::COLS + col
}

pub fn action_2_coord(action: usize) -> (usize, usize){
    let row = action / PenteState::COLS;
    let col = action - row * PenteState::COLS;
    (row, col)
}



impl fmt::Display for PenteState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "   ")?;
        for col in 0..PenteState::COLS{
            write!(f, "{:02} ", col)?;
        }
        writeln!(f)?;
        for row in 0..PenteState::ROWS{
            write!(f, "{:02}  ", row)?;
            for col in 0..PenteState::COLS{
                if self.board_[row][col] == Piece::Empty{
                    write!(f, "·  ")?;
                }else if self.board_[row][col]  == Piece::Black{
                    write!(f, "B  ")?;
                }else{
                    write!(f, "W  ")?;
                }
            }
            writeln!(f)?;
            }
        writeln!(f, "Black captures:{}   White Captures:{}", self.b_caps_, self.w_caps_)?;
        Ok(())
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    // Builds a position from rows of '.', 'B', 'W' (missing rows/cols are empty).
    fn board(rows: &[&str], black_to_move: bool) -> PenteState{
        let mut s = PenteState::new();
        for (r, row) in rows.iter().enumerate(){
            for (c, ch) in row.chars().enumerate(){
                s.board_[r][c] = match ch{
                    'B' => Piece::Black,
                    'W' => Piece::White,
                    _ => Piece::Empty,
                };
                if s.board_[r][c] != Piece::Empty{
                    s.num_pieces_ += 1;
                }
            }
        }
        s.is_black_ = black_to_move;
        return s;
    }

    fn piece(s: &PenteState, r: usize, c: usize) -> Piece{
        return s.board_[r][c];
    }

    #[test]
    fn new_game_is_empty_with_black_to_move(){
        let s = PenteState::new();
        assert!(s.is_black());
        assert!(!s.is_terminal());
        assert!(s.legal_actions_mask().iter().all(|&l| l));
        assert_eq!(s.legal_actions_mask().len(), ROWS * COLS);
    }

    #[test]
    fn play_alternates_players_and_rejects_occupied_squares(){
        let mut s = PenteState::new();
        assert!(s.play(4, 4));
        assert_eq!(piece(&s, 4, 4), Piece::Black);
        assert!(!s.is_black());
        assert!(!s.play(4, 4), "occupied square must be rejected");
        assert!(!s.is_black(), "a rejected move must not change the side to move");
        assert!(s.play(4, 5));
        assert_eq!(piece(&s, 4, 5), Piece::White);
        assert!(!s.is_legal_action(coord_2_action(4, 4)));
    }

    #[test]
    fn play_rejects_out_of_bounds_without_wrapping(){
        let mut s = PenteState::new();
        // (0, COLS) would flatten to the action for (1, 0).
        assert!(!s.play(0, COLS));
        assert!(!s.play(ROWS, 0));
        assert_eq!(piece(&s, 1, 0), Piece::Empty);
        assert!(s.is_black());
        assert!(!s.is_legal_action(ROWS * COLS));
    }

    #[test]
    fn action_coord_roundtrip(){
        for a in 0..ROWS * COLS{
            let (r, c) = action_2_coord(a);
            assert!(r < ROWS && c < COLS);
            assert_eq!(coord_2_action(r, c), a);
        }
    }

    // Black completes five by filling (r, c); checks a win in that direction.
    fn assert_black_wins_by_playing(rows: &[&str], r: usize, c: usize){
        let mut s = board(rows, true);
        assert!(s.play(r, c));
        assert!(s.is_terminal(), "expected a win after playing ({r}, {c})\n{s}");
        assert_eq!(s.get_terminal_value(), -1.0, "side to move after a win has lost");
        assert!(s.legal_actions_mask().iter().all(|&l| !l));
    }

    #[test]
    fn five_in_a_row_wins_in_every_direction(){
        // horizontal, filling the middle of the line
        assert_black_wins_by_playing(&["", "", "..BB.BB.."], 2, 4);
        // vertical, filling the end
        assert_black_wins_by_playing(&[".B", ".B", ".B", ".B"], 4, 1);
        // diagonal
        assert_black_wins_by_playing(&["B", ".B", "..B", "...B"], 4, 4);
        // anti-diagonal
        assert_black_wins_by_playing(&["....B", "...B", "..B", ".B"], 4, 0);
    }

    #[test]
    fn overline_of_six_also_wins(){
        assert_black_wins_by_playing(&["BBB.BB"], 0, 3);
    }

    #[test]
    fn four_in_a_row_or_a_broken_line_does_not_win(){
        let mut s = board(&["BBB"], true);
        assert!(s.play(0, 3));
        assert!(!s.is_terminal());
        // An opponent stone breaks the line.
        let mut s = board(&["BBWB."], true);
        assert!(s.play(0, 4));
        assert!(!s.is_terminal());
    }

    #[test]
    fn captures_a_flanked_pair_in_every_direction(){
        const DIRS: [(i32, i32); 8] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)];
        for (dr, dc) in DIRS{
            // Black plays at the centre; W W then B lie along the direction.
            let (r0, c0) = (4i32, 4i32);
            let mut s = PenteState::new();
            s.board_[(r0 + dr) as usize][(c0 + dc) as usize] = Piece::White;
            s.board_[(r0 + 2 * dr) as usize][(c0 + 2 * dc) as usize] = Piece::White;
            s.board_[(r0 + 3 * dr) as usize][(c0 + 3 * dc) as usize] = Piece::Black;
            s.num_pieces_ = 3;
            assert!(s.play(4, 4));
            assert_eq!(s.b_caps_, 1, "direction ({dr}, {dc})");
            assert_eq!(piece(&s, (r0 + dr) as usize, (c0 + dc) as usize), Piece::Empty);
            assert_eq!(piece(&s, (r0 + 2 * dr) as usize, (c0 + 2 * dc) as usize), Piece::Empty);
            assert_eq!(s.num_pieces_, 2, "two stones removed, one added");
        }
    }

    #[test]
    fn one_move_can_capture_two_pairs(){
        let mut s = board(&["BWW.WWB"], true);
        assert!(s.play(0, 3));
        assert_eq!(s.b_caps_, 2);
        assert_eq!(s.num_pieces_, 3);
    }

    #[test]
    fn three_stones_or_unflanked_pairs_are_not_captured(){
        let mut s = board(&["BWWW."], true);
        assert!(s.play(0, 4));
        assert_eq!(s.b_caps_, 0);
        let mut s = board(&[".WW."], true);
        assert!(s.play(0, 3));
        assert_eq!(s.b_caps_, 0);
    }

    #[test]
    fn moving_into_a_flanked_position_is_safe(){
        // White completes B W W B itself: no capture.
        let mut s = board(&["BW.B"], false);
        assert!(s.play(0, 2));
        assert_eq!(s.b_caps_ + s.w_caps_, 0);
        assert_eq!(piece(&s, 0, 1), Piece::White);
        assert_eq!(piece(&s, 0, 2), Piece::White);
    }

    #[test]
    fn fifth_capture_wins(){
        // White already has 4 captures and captures a fifth pair.
        let mut s = board(&["WBB."], false);
        s.w_caps_ = 4;
        assert!(s.play(0, 3));
        assert_eq!(s.w_caps_, 5);
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), -1.0);
        // A fourth capture is not enough.
        let mut s = board(&["WBB."], false);
        s.w_caps_ = 3;
        assert!(s.play(0, 3));
        assert_eq!(s.w_caps_, 4);
        assert!(!s.is_terminal());
    }

    #[test]
    fn full_board_without_a_win_is_a_draw(){
        // Pattern with no five in any direction (runs of at most 2), with
        // (8, 8) left empty. White's stone there makes no five and no capture.
        let mut s = PenteState::new();
        for r in 0..ROWS{
            for c in 0..COLS{
                s.board_[r][c] = if (c + 2 * r) % 4 < 2 {Piece::Black} else {Piece::White};
            }
        }
        s.board_[8][8] = Piece::Empty;
        s.num_pieces_ = ROWS * COLS - 1;
        s.is_black_ = false;
        assert!(s.play(8, 8));
        assert!(s.is_terminal());
        assert_eq!(s.get_terminal_value(), 0.0);
    }

    #[test]
    fn finished_game_rejects_further_moves(){
        let mut s = board(&["BBBB"], true);
        assert!(s.play(0, 4));
        assert!(!s.play(5, 5));
        assert!(!s.is_legal_action(coord_2_action(5, 5)));
    }

    #[test]
    #[should_panic(expected = "finished game")]
    fn apply_action_panics_after_game_over(){
        let mut s = board(&["BBBB"], true);
        s.apply_action(coord_2_action(0, 4));
        s.apply_action(coord_2_action(5, 5));
    }

    #[test]
    #[should_panic(expected = "Illegal action")]
    fn apply_action_panics_on_occupied_square(){
        let mut s = PenteState::new();
        s.apply_action(0);
        s.apply_action(0);
    }

    #[test]
    #[should_panic(expected = "non-terminal")]
    fn terminal_value_panics_on_ongoing_game(){
        PenteState::new().get_terminal_value();
    }

    #[test]
    fn to_tensor_is_from_the_side_to_moves_perspective(){
        let plane = ROWS * COLS;
        let at = |chan: usize, r: usize, c: usize| chan * plane + r * COLS + c;
        let mut s = PenteState::new();
        s.play(4, 4); // Black stone; White to move
        let t = s.to_tensor();
        assert_eq!(t.len(), PenteState::N_CHAN * plane);
        assert_eq!(t[at(0, 4, 4)], 0.0, "channel 0 is the side to move (White)");
        assert_eq!(t[at(2, 4, 4)], 1.0, "channel 2 is the opponent (Black)");
        // Captures fill the whole plane of the capturing player's channel.
        let mut s = board(&["BWW."], true);
        s.play(0, 3); // Black captures; White to move
        let t = s.to_tensor();
        assert!(t[at(3, 0, 0)..at(4, 0, 0)].iter().all(|&x| x == 1.0), "opponent captures");
        assert!(t[at(1, 0, 0)..at(2, 0, 0)].iter().all(|&x| x == 0.0), "own captures");
    }
}
