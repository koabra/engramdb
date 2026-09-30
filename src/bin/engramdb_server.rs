use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use engramdb::{serve_flight, Engine, SessionManager};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("engramdb-server: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let data_dir = option(&arguments, "--data-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./engramdb-data"));
    let address: SocketAddr = option(&arguments, "--address")
        .unwrap_or_else(|| "127.0.0.1:50051".to_owned())
        .parse()?;
    let engine = Arc::new(Engine::open(data_dir)?);
    let main = engine.main_branch().id;
    let sessions = Arc::new(SessionManager::new(engine));
    println!("EngramDB Flight listening on {address}; main_branch={main}");
    serve_flight(address, sessions).await?;
    Ok(())
}

fn option(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}
