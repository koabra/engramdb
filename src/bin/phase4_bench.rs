use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use engramdb::{
    HardwareCapabilities, Hash, KvCacheSpec, KvCacheStore, KvDType, KvLayout, TransferPath,
};
use uuid::Uuid;

fn main() {
    if let Err(error) = run() {
        eprintln!("phase4-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str).unwrap_or("help") {
        "capability" => capability(&arguments[1..]),
        "bandwidth" => bandwidth(&arguments[1..]),
        "ttft-sim" => ttft_sim(&arguments[1..]),
        _ => {
            eprintln!(
                "usage:\n  phase4-bench capability --output JSON\n  \
                 phase4-bench bandwidth --data-dir PATH --output CSV \
                 [--bytes 134217728] [--iterations 3]\n  \
                 phase4-bench ttft-sim --data-dir PATH --output CSV \
                 [--tokens 1000,10000] [--bytes-per-token 4096]"
            );
            Ok(())
        }
    }
}

fn capability(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let output = required_path(arguments, "--output")?;
    output_writer(&output)?.write_all(HardwareCapabilities::detect().to_json()?.as_bytes())?;
    Ok(())
}

fn bandwidth(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let bytes = usize_option(arguments, "--bytes", 128 * 1024 * 1024)?;
    let iterations = usize_option(arguments, "--iterations", 3)?;
    reset_directory(&data_dir)?;
    let store = KvCacheStore::open(&data_dir)?;
    let spec = spec_for_bytes(bytes)?;
    let payload: Vec<u8> = (0..bytes).map(|index| (index % 251) as u8).collect();
    let capabilities = HardwareCapabilities::detect();
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "operation,backend,hardware_verified,bytes,iterations,elapsed_s,gb_per_s,cpu_s,cpu_pct,cache_hash"
    )?;

    let cpu_before = process_cpu_seconds()?;
    let started = Instant::now();
    let mut manifest = None;
    for _ in 0..iterations {
        manifest = Some(store.put(Uuid::new_v4(), spec, &payload)?);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let cpu = process_cpu_seconds()? - cpu_before;
    writeln!(
        writer,
        "write,cpu-direct,false,{bytes},{iterations},{elapsed:.6},{:.6},{cpu:.6},{:.3},{}",
        bytes as f64 * iterations as f64 / elapsed / 1_000_000_000.0,
        cpu / elapsed * 100.0,
        manifest.unwrap().cache_hash
    )?;

    let branch = Uuid::new_v4();
    let manifest = store.put(branch, spec, &payload)?;
    let cpu_before = process_cpu_seconds()?;
    let started = Instant::now();
    for _ in 0..iterations {
        let restored = store.get(branch)?.expect("benchmark cache exists");
        std::hint::black_box(restored.bytes);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let cpu = process_cpu_seconds()? - cpu_before;
    writeln!(
        writer,
        "read,cpu-direct,false,{bytes},{iterations},{elapsed:.6},{:.6},{cpu:.6},{:.3},{}",
        bytes as f64 * iterations as f64 / elapsed / 1_000_000_000.0,
        cpu / elapsed * 100.0,
        manifest.cache_hash
    )?;

    if capabilities.transfer_path == TransferPath::GdsDirectVerified {
        return Err("verified GDS path requires the hardware-only gdsio runner".into());
    }
    Ok(())
}

fn ttft_sim(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let tokens = option(arguments, "--tokens")
        .unwrap_or_else(|| "1000,10000".to_owned())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    let bytes_per_token = usize_option(arguments, "--bytes-per-token", 4096)?;
    reset_directory(&data_dir)?;
    let store = KvCacheStore::open(&data_dir)?;
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "mode,hardware_verified,tokens,cache_bytes,elapsed_ms,checksum"
    )?;
    for token_count in tokens {
        let mut accumulator = 0_f32;
        let started = Instant::now();
        for token in 0..token_count {
            for dimension in 0..256 {
                accumulator += ((token * 31 + dimension * 17) % 251) as f32 * 0.0001;
            }
        }
        writeln!(
            writer,
            "prefill-cpu-simulation,false,{token_count},{},{:.6},{}",
            token_count * bytes_per_token,
            started.elapsed().as_secs_f64() * 1000.0,
            accumulator.to_bits()
        )?;

        let bytes = token_count
            .checked_mul(bytes_per_token)
            .ok_or("simulated KV byte count overflow")?;
        let spec = spec_for_bytes(bytes)?;
        let payload = vec![0x5a; bytes];
        let branch = Uuid::new_v4();
        store.put(branch, spec, &payload)?;
        let started = Instant::now();
        let restored = store.get(branch)?.expect("simulated cache exists");
        writeln!(
            writer,
            "kv-restore-cpu-direct,false,{token_count},{bytes},{:.6},{}",
            started.elapsed().as_secs_f64() * 1000.0,
            restored.manifest.cache_hash
        )?;
    }
    Ok(())
}

fn spec_for_bytes(bytes: usize) -> Result<KvCacheSpec, Box<dyn std::error::Error>> {
    const BYTES_PER_BLOCK: usize = 2 * 16 * 16 * 2;
    if bytes == 0 || bytes % BYTES_PER_BLOCK != 0 {
        return Err(format!("--bytes must be a non-zero multiple of {BYTES_PER_BLOCK}").into());
    }
    let num_blocks = u32::try_from(bytes / BYTES_PER_BLOCK)?;
    Ok(KvCacheSpec {
        model_fingerprint: Hash(*blake3::hash(b"phase4-benchmark-model").as_bytes()),
        dtype: KvDType::Fp16,
        layout: KvLayout::VllmPagedV1,
        num_layers: 1,
        num_blocks,
        num_kv_heads: 1,
        head_size: 16,
        block_size: 16,
        sequence_length: num_blocks * 16,
        tensor_parallel_rank: 0,
        tensor_parallel_world: 1,
    })
}

fn process_cpu_seconds() -> io::Result<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the provided rusage on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful getrusage initialized the value.
    let usage = unsafe { usage.assume_init() };
    Ok(timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime))
}

fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
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

fn usize_option(
    arguments: &[String],
    name: &str,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(option(arguments, name)
        .unwrap_or_else(|| default.to_string())
        .parse()?)
}

fn required_path(arguments: &[String], name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    option(arguments, name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing required option {name}").into())
}
