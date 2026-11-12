use std::fmt::{self, Debug};

pub type Action = usize;

// GameState is an abstraction of a 2D board game.
pub trait GameState: Default + Clone + fmt::Display + Debug + Send + 'static{
    // Short unique identifier (e.g. "minipente"). Saved with every model so a
    // model trained for one game can't be loaded as another.
    const NAME: &'static str;
    const ACTN_SPC_SIZE: usize; // Input action space is scalar [0, ACTN_SPC_SIZE).
    const N_CHAN: usize; // Game can be represented as flattened tensor [[[f32; COLS]; ROWS]; N_CHAN]
    const ROWS: usize;
    const COLS: usize;

    fn to_tensor(&self) -> Vec<f32>; // flattened N_CHAN x ROWS x COLS

    // One entry per action (length ACTN_SPC_SIZE): true = legal.
    fn legal_actions_mask(&self) -> Vec<bool>;

    // Apply an action. action MUST be a legal action.
    // Implementations should panic on an illegal action or a finished game.
    fn apply_action(&mut self, action: Action);

    // Is the game over
    fn is_terminal(&self) -> bool;

    // Must only be called on a terminal state.
    // Returns:
    //  +1 on Current player win
    //   0 on Draw
    //  -1 on Opponent win
    // note for games like chess will always be 0 or -1 because
    // game ends when current player has no moves (draw or checkmated). 
    fn get_terminal_value(&self) -> f32;
}