use std::{fmt};
use tracing::trace;

use crate::gamestate::GameState;

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

impl Default for PenteState{
    fn default() -> Self {
        return Self::new();
    }
}

impl GameState for PenteState{
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


