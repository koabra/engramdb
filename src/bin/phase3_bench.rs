use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use engramdb::{optimize, parse_enql, plan_logical, CatalogStats, Engine, SessionManager};

fn main() {
    if let Err(error) = run() {
        eprintln!("phase3-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str).unwrap_or("help") {
        "sessions" => sessions(&arguments[1..]),
        "planner" => planner(&arguments[1..]),
        _ => {
            eprintln!(
                "usage:\n  phase3-bench sessions --data-dir PATH --output CSV \
                 [--levels 1,10,100,1000]\n  \
                 phase3-bench planner --output CSV [--iterations 100000]"
            );
            Ok(())
        }
    }
}

fn sessions(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let levels = option(arguments, "--levels")
        .unwrap_or_else(|| "1,10,100,1000".to_owned())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if levels.is_empty() || !levels.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("session levels must be strictly increasing".into());
    }
    reset_directory(&data_dir)?;
    let engine = Arc::new(Engine::open(&data_dir)?);
    let manager = SessionManager::new(Arc::clone(&engine));
    let main = engine.main_branch().id;
    let mut current = 0;
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,sessions,batch_sessions,elapsed_s,rss_bytes,physical_bytes"
    )?;
    for level in levels {
        let started = Instant::now();
        for _ in current..level {
            manager.fork_session(main)?;
        }
        let elapsed = started.elapsed().as_secs_f64();
        writeln!(
            writer,
            "engramdb,{level},{},{elapsed:.6},{},{}",
            level - current,
            resident_bytes()?,
            directory_size(&data_dir)?
        )?;
        current = level;
    }
    Ok(())
}

fn planner(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let output = required_path(arguments, "--output")?;
    let iterations = option(arguments, "--iterations")
        .unwrap_or_else(|| "100000".to_owned())
        .parse::<usize>()?;
    let hash = blake3::hash(b"planner-root").to_hex();
    let graph_query = format!(
        "MATCH (a)-[:EDGE*1..3]->(b) FROM {hash} VECTOR [1.0,0.0] \
         COSINE > 0.8 ASSERTION < 100 VALID <= 20 EDGE_TYPE = 7 \
         AS OF SYSTEM TIME 99 RETURN id,score,depth LIMIT 10"
    );
    let vector_query = format!(
        "MATCH (a)-[:EDGE*1..3]->(b) FROM {hash} VECTOR [1.0,0.0] \
         COSINE > 0.99 ASSERTION < 100 VALID <= 20 \
         AS OF SYSTEM TIME 99 RETURN id,score,depth LIMIT 10"
    );
    let mut writer = output_writer(&output)?;
    writeln!(writer, "plan,iterations,elapsed_ns,plans_per_s")?;
    for (name, query) in [("graph_first", graph_query), ("vector_first", vector_query)] {
        let started = Instant::now();
        for _ in 0..iterations {
            let parsed = parse_enql(&query)?;
            let logical = plan_logical(&parsed);
            std::hint::black_box(optimize(
                logical,
                CatalogStats {
                    nodes: 1_000_000,
                    average_out_degree: 8.0,
                },
            ));
        }
        let elapsed = started.elapsed();
        writeln!(
            writer,
            "{name},{iterations},{},{:.3}",
            elapsed.as_nanos(),
            iterations as f64 / elapsed.as_secs_f64()
        )?;
    }
    Ok(())
}

fn resident_bytes() -> io::Result<u64> {
    let mut status = String::new();
    File::open("/proc/self/status")?.read_to_string(&mut status)?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .ok_or_else(|| io::Error::other("VmRSS is absent from /proc/self/status"))?;
    let kib = line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| io::Error::other("VmRSS has no value"))?
        .parse::<u64>()
        .map_err(io::Error::other)?;
    Ok(kib * 1024)
}

fn directory_size(path: &Path) -> io::Result<u64> {
    fs::read_dir(path)?.try_fold(0_u64, |total, entry| {
        let metadata = entry?.metadata()?;
        Ok(total
            + if metadata.is_file() {
                metadata.len()
            } else {
                0
            })
    })
}

fn reset_directory(path: &Path) -> io::Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    fs::create_dir_all(path)
}

fn output_writer(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    File::create(path)
}

fn option(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}

fn required_path(arguments: &[String], name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    option(arguments, name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing required option {name}").into())
}
