//! The `tard` subcommands, one module each. Each has an `Args` (its flags)
//! and a `run(args)`.
pub mod arena;
pub mod play;
pub mod train;
pub mod web;

use std::path::Path;

use tard::games::Game;
use tard::model::read_model_meta;

// The game a model file was trained for, from its .net.json sidecar.
// Panics (with the reason) if it can't be read or names an unknown game.
pub fn game_of_model(model: &Path) -> Game {
    let meta = read_model_meta(model)
        .unwrap_or_else(|e| panic!("Failed to read model {}: {}", model.display(), e));
    return Game::from_name(&meta.game).unwrap_or_else(|| {
        panic!(
            "Model {} is for game '{}', which this build doesn't know",
            model.display(),
            meta.game
        )
    });
}
