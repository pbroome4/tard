//! The games Tard can play: the `GameState` interface the engine, model and
//! trainer are generic over, the `TextGame` interface the commands use to
//! show and read moves, `WebGame` for the web page to draw them, and `Game`,
//! the one list of every implemented game.
//!
//! Each game is a separate, self-contained module. To add one: implement
//! `GameState`, `TextGame` and `WebGame` in a new module, then add a `Game` variant and
//! an arm to `with_game!` and `Game::default_net` below.

use std::fmt::{self, Debug};
use clap::ValueEnum;
use serde::Serialize;

use crate::model::NetConfig;

pub mod minipente;
pub mod pente;
pub mod hive;

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

// What the commands (play, arena) need beyond GameState: player names, and
// moves as text a person can read and type.
pub trait TextGame: GameState{
    // Names of the first and second player, e.g. ["Black", "White"].
    const PLAYERS: [&'static str; 2];

    // How to type a move, shown in prompts, e.g. "row, col".
    const MOVE_FORMAT: &'static str;

    // Name of the player to move.
    fn active_player(&self) -> &'static str;

    // Parses a typed move. Errors, saying why, unless it is a legal move here.
    fn parse_action(&self, input: &str) -> Result<Action, String>;

    // `action` (legal here) as text in the format parse_action reads.
    fn action_text(&self, action: Action) -> String;
}

// What `tard web` needs beyond TextGame: the position and moves as JSON for
// the page to draw. Each game picks its own shape, tagged by "kind";
// web/index.html knows how to draw each kind.
pub trait WebGame: TextGame{
    // The whole position: pieces, captures or hands, and so on.
    fn position_json(&self) -> serde_json::Value;

    // Where `action` (legal here) plays, in the position's terms.
    fn action_json(&self, action: Action) -> serde_json::Value;
}

/// Every implemented game. Its value names are the `GameState::NAME`s, used
/// for `--game` and saved in config.json and model sidecars.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Game{
    /// Pente on a 9x9 board, no opening restrictions
    #[value(name = "minipente")]
    MiniPente,
    /// Official Pente: 19x19, White first, tournament opening rule
    Pente,
    /// Hive, base game (no expansion pieces)
    Hive,
}

/// Runs `$body` with `$G` naming the state type of `$game` (a `Game`), so
/// generic code can be called for whichever game was picked at runtime:
/// `with_game!(game, G => train::<G>(&args))`.
#[macro_export]
macro_rules! with_game{
    ($game:expr, $G:ident => $body:expr) => {
        match $game{
            $crate::games::Game::MiniPente => { type $G = $crate::games::minipente::MiniPenteState; $body }
            $crate::games::Game::Pente => { type $G = $crate::games::pente::PenteState; $body }
            $crate::games::Game::Hive => { type $G = $crate::games::hive::HiveState; $body }
        }
    };
}

impl Game{
    pub const ALL: [Game; 3] = [Game::MiniPente, Game::Pente, Game::Hive];

    /// The game's `GameState::NAME`.
    pub fn name(self) -> &'static str{
        return with_game!(self, G => G::NAME);
    }

    /// The game whose `GameState::NAME` is `name`.
    pub fn from_name(name: &str) -> Option<Game>{
        return Self::ALL.into_iter().find(|g| g.name() == name);
    }

    /// Network size for a fresh model of this game.
    pub fn default_net(self) -> NetConfig{
        return match self{
            Game::MiniPente => minipente::default_net_config(),
            Game::Pente => pente::default_net_config(),
            Game::Hive => hive::default_net_config(),
        };
    }
}

/// Parses "row, col" (spaces optional), as the grid games type their moves.
pub fn parse_coord(s: &str) -> Result<(usize, usize), String> {
    let mut parts = s.trim().split(',');
    if let (Some(row_str), Some(col_str)) = (parts.next(), parts.next()) {
        if let (Ok(row), Ok(col)) = (row_str.trim().parse(), col_str.trim().parse()) {
            return Ok((row, col));
        }
    }
    return Err(String::from("Invalid format"));
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn game_names_match_their_states_and_round_trip(){
        assert_eq!(Game::ALL.map(Game::name), ["minipente", "pente", "hive"]);
        for g in Game::ALL{
            assert_eq!(Game::from_name(g.name()), Some(g));
            assert_eq!(g.to_possible_value().unwrap().get_name(), g.name(), "--game value");
            assert_eq!(serde_json::to_string(&g).unwrap(), format!("\"{}\"", g.name()), "config.json value");
        }
        assert_eq!(Game::from_name("chess"), None);
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
}
