use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

use engramdb::{Engine, TemporalRecord};

fn main() {
    if let Err(error) = run() {
        eprintln!("phase1-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let command = arguments.first().map(String::as_str).unwrap_or("help");
    match command {
        "branch" => branch_benchmark(&arguments[1..]),
        "write" => write_benchmark(&arguments[1..]),
        _ => {
            eprintln!(
                "usage:\n  phase1-bench branch --data-dir PATH --output CSV \
                 [--concurrency 1,10,100,1000] [--seed-records 1000]\n  \
                 phase1-bench write --data-dir PATH --output CSV \
                 [--workers 5000] [--physical-threads 4]"
            );
            Ok(())
        }
    }
}

fn branch_benchmark(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let concurrency = option(arguments, "--concurrency")
        .unwrap_or_else(|| "1,10,100,1000".to_owned())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    let seed_records = option(arguments, "--seed-records")
        .unwrap_or_else(|| "1000".to_owned())
        .parse::<usize>()?;
    reset_directory(&data_dir)?;
    let engine = Arc::new(Engine::open(&data_dir)?);
    let main = engine.main_branch().id;
    if seed_records > 0 {
        let mut transaction = engine.begin(main)?;
        for index in 0..seed_records {
            transaction.put(TemporalRecord::new(
                format!("seed-{index:08}"),
                vec![(index % 251) as u8; 512],
                0,
                1_000_000,
            )?)?;
        }
        transaction.commit()?;
    }

    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,operation,concurrency,sample,latency_us,batch_us,dataset_records"
    )?;
    for concurrent in concurrency {
        let barrier = Arc::new(Barrier::new(concurrent + 1));
        let results = Arc::new(Mutex::new(Vec::with_capacity(concurrent)));
        let batch_start = Instant::now();
        std::thread::scope(|scope| {
            for sample in 0..concurrent {
                let barrier = Arc::clone(&barrier);
                let results = Arc::clone(&results);
                let engine = Arc::clone(&engine);
                std::thread::Builder::new()
                    .name(format!("fork-{sample}"))
                    .stack_size(128 * 1024)
                    .spawn_scoped(scope, move || {
                        barrier.wait();
                        let started = Instant::now();
                        engine.fork(main).expect("branch benchmark fork");
                        let elapsed = started.elapsed().as_secs_f64() * 1_000_000.0;
                        results.lock().unwrap().push((sample, elapsed));
                    })
                    .expect("spawn branch benchmark worker");
            }
            barrier.wait();
        });
        let batch_us = batch_start.elapsed().as_secs_f64() * 1_000_000.0;
        let mut results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        results.sort_by_key(|(sample, _)| *sample);
        for (sample, latency_us) in results {
            writeln!(
                writer,
                "engramdb,fork,{concurrent},{sample},{latency_us:.3},{batch_us:.3},{seed_records}"
            )?;
        }
    }
    Ok(())
}

fn write_benchmark(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let workers = option(arguments, "--workers")
        .unwrap_or_else(|| "5000".to_owned())
        .parse::<usize>()?;
    let physical_threads = option(arguments, "--physical-threads")
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .to_string()
        })
        .parse::<usize>()?
        .max(1);
    reset_directory(&data_dir)?;
    let engine = Arc::new(Engine::open(&data_dir)?);
    let main = engine.main_branch().id;
    let mut branches = Vec::with_capacity(workers);
    for _ in 0..workers {
        branches.push(engine.fork(main)?.id);
    }
    let branches = Arc::new(branches);
    let next = Arc::new(AtomicUsize::new(0));
    let physical_before = directory_size(&data_dir)?;
    let logical_bytes = workers as u64 * 1024;
    let started = Instant::now();
    std::thread::scope(|scope| {
        for thread_id in 0..physical_threads {
            let engine = Arc::clone(&engine);
            let branches = Arc::clone(&branches);
            let next = Arc::clone(&next);
            scope.spawn(move || loop {
                let worker = next.fetch_add(1, Ordering::Relaxed);
                if worker >= branches.len() {
                    break;
                }
                let mut transaction = engine.begin(branches[worker]).expect("begin write");
                transaction
                    .put(
                        TemporalRecord::new(
                            format!("worker-{worker:08}"),
                            vec![(worker % 251) as u8; 1024],
                            0,
                            1_000_000,
                        )
                        .expect("record"),
                    )
                    .expect("stage write");
                transaction.commit().expect("commit write");
                std::hint::black_box(thread_id);
            });
        }
    });
    let elapsed = started.elapsed().as_secs_f64();
    let physical_after = directory_size(&data_dir)?;
    let physical_bytes = physical_after - physical_before;
    let amplification = physical_bytes as f64 / logical_bytes as f64;
    let ops_per_second = workers as f64 / elapsed;
    let io = engine.io_stats();

    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,operation,logical_workers,physical_threads,writes,record_bytes,elapsed_s,ops_per_s,logical_bytes,physical_bytes,write_amplification,io_page_bytes"
    )?;
    writeln!(
        writer,
        "engramdb,micro_write,{workers},{physical_threads},{workers},1024,{elapsed:.6},{ops_per_second:.3},{logical_bytes},{physical_bytes},{amplification:.6},{}",
        io.bytes_written
    )?;
    Ok(())
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
