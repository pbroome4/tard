//! The `tard` command line: train models by self-play and play against them.
use clap::{Parser, Subcommand};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};

mod commands;

#[derive(Parser)]
#[command(
    name = "tard",
    about = "AlphaZero-style self-play AI for board games (MiniPente, Pente, Hive, Ultimate Tic-Tac-Toe)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a model by self-play
    Train(commands::train::Args),
    /// Play in the terminal: against a model (--model), or human vs human
    Play(commands::play::Args),
    /// Play any game in your web browser: against a model, a friend, or model vs model
    Web(commands::web::Args),
    /// Play two models against each other and estimate their Elo difference
    Arena(commands::arena::Args),
}

fn main() {
    let cli = Cli::parse();
    // Warnings from dependencies, info from Tard; RUST_LOG adds to this
    // (e.g. RUST_LOG=tard=debug).
    let filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::WARN.into())
        .add_directive("tard=info".parse().unwrap());
    tracing_subscriber::fmt().with_env_filter(filter).init();
    match cli.command {
        Command::Train(args) => commands::train::run(args),
        Command::Play(args) => commands::play::run(args),
        Command::Web(args) => commands::web::run(args),
        Command::Arena(args) => commands::arena::run(args),
    }
}
