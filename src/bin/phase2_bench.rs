use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use engramdb::{
    dot_i8, selected_simd_flavor, AlignedBlock, DirectIo, DistanceMetric, FusedBlockBuilder,
    FusedBlockView, GraphEdge, HnswConfig, HybridIndex, HybridRecord, FUSED_BLOCK_SIZE,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("phase2-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str).unwrap_or("help") {
        "sift" => sift(&arguments[1..]),
        "query" => query(&arguments[1..]),
        "storage" => storage(&arguments[1..]),
        "profile-fused" => profile_fused(&arguments[1..]),
        "profile-split" => profile_split(&arguments[1..]),
        _ => {
            eprintln!(
                "usage:\n  phase2-bench sift --sift-dir PATH --data-dir PATH --output CSV \
                 [--base-limit 1000000] [--queries 10000]\n  \
                 phase2-bench query --data-dir PATH --output CSV \
                 [--nodes 10000] [--queries 10000] [--dimension 128]\n  \
                 phase2-bench storage --data-dir PATH --output CSV \
                 [--sample-nodes 100000] [--dimension 768] [--edges 10]\n  \
                 phase2-bench profile-fused --data-dir PATH\n  \
                 phase2-bench profile-split"
            );
            Ok(())
        }
    }
}

fn sift(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let sift_dir = required_path(arguments, "--sift-dir")?;
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let base_limit = usize_option(arguments, "--base-limit", 1_000_000)?;
    let query_limit = usize_option(arguments, "--queries", 10_000)?;
    reset_directory(&data_dir)?;

    let base = read_fvecs(&sift_dir.join("sift_base.fvecs"), base_limit)?;
    let queries = read_fvecs(&sift_dir.join("sift_query.fvecs"), query_limit)?;
    if base.is_empty() || queries.is_empty() || base[0].len() != queries[0].len() {
        return Err("SIFT vectors are empty or dimensions differ".into());
    }
    let dimension = base[0].len();
    let records = base
        .iter()
        .enumerate()
        .map(|(id, vector)| HybridRecord {
            id: id as u64,
            assertion_time: 1,
            valid_from: i64::MIN,
            valid_to: i64::MAX,
            vector: vector.clone(),
            edges: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut index = HybridIndex::open(
        &data_dir,
        dimension,
        DistanceMetric::L2,
        HnswConfig {
            max_connections: 16,
            ef_construction: 128,
            ef_search: 1024,
        },
    )?;
    let build_started = Instant::now();
    index.insert(1, records)?;
    let build_s = build_started.elapsed().as_secs_f64();

    let groundtruth_path = sift_dir.join("sift_groundtruth.ivecs");
    let published_truth = if base.len() == 1_000_000 && groundtruth_path.exists() {
        Some(read_ivecs(&groundtruth_path, queries.len())?)
    } else {
        None
    };
    let mut recalled = 0_usize;
    let mut expected_total = 0_usize;
    let query_started = Instant::now();
    for (query_index, query) in queries.iter().enumerate() {
        let expected = if let Some(truth) = &published_truth {
            truth[query_index]
                .iter()
                .take(10)
                .map(|value| *value as u64)
                .collect::<HashSet<_>>()
        } else {
            exact_l2(&base, query, 10)
        };
        let actual = index
            .nearest(query, 10, 0, u64::MAX)?
            .hits
            .into_iter()
            .map(|hit| hit.id)
            .collect::<HashSet<_>>();
        recalled += expected.intersection(&actual).count();
        expected_total += expected.len();
    }
    let query_s = query_started.elapsed().as_secs_f64();
    let recall = recalled as f64 / expected_total as f64;
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "dataset,base_vectors,queries,dimension,build_s,query_s,recall_at_10,simd,disk_bytes,truth"
    )?;
    writeln!(
        writer,
        "SIFT1M,{},{},{dimension},{build_s:.6},{query_s:.6},{recall:.6},{:?},{},{}",
        base.len(),
        queries.len(),
        selected_simd_flavor(),
        index.disk_bytes(),
        if published_truth.is_some() {
            "published"
        } else {
            "exact_subset"
        }
    )?;
    if recall <= 0.95 {
        return Err(format!("Recall@10 {recall:.4} did not exceed 0.95").into());
    }
    Ok(())
}

fn query(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let nodes = usize_option(arguments, "--nodes", 10_000)?;
    let query_count = usize_option(arguments, "--queries", 10_000)?;
    let dimension = usize_option(arguments, "--dimension", 128)?;
    reset_directory(&data_dir)?;
    let mut index = HybridIndex::open(
        &data_dir,
        dimension,
        DistanceMetric::Cosine,
        HnswConfig::default(),
    )?;
    let mut records = synthetic_records(nodes, dimension, 10);
    for id in (0..nodes).step_by(10) {
        let mut revision = records[id].clone();
        revision.assertion_time = 20;
        revision.vector.fill(0.0);
        revision.vector[(id + 1) % dimension.min(32)] = 1.0;
        records.push(revision);
    }
    index.insert(1, records)?;
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,query,latency_us,hits,visited_records,physical_block_reads,nodes,dimension,hops"
    )?;
    for query_id in 0..query_count {
        let root = (query_id.wrapping_mul(7919) % nodes) as u64;
        let mut vector = vec![0.0; dimension];
        vector[root as usize % dimension.min(32)] = 1.0;
        let started = Instant::now();
        let result = index.traverse(root, &vector, 3, 0.8, 500, 10, |_| true)?;
        let latency_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        writeln!(
            writer,
            "engramdb,{query_id},{latency_us:.3},{},{},{},{nodes},{dimension},3",
            result.hits.len(),
            result.stats.visited_records,
            result.stats.physical_block_reads
        )?;
    }
    Ok(())
}

fn storage(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = required_path(arguments, "--data-dir")?;
    let output = required_path(arguments, "--output")?;
    let sample_nodes = usize_option(arguments, "--sample-nodes", 100_000)?;
    let dimension = usize_option(arguments, "--dimension", 768)?;
    let edge_count = usize_option(arguments, "--edges", 10)?;
    reset_directory(&data_dir)?;
    let io = DirectIo::<FUSED_BLOCK_SIZE>::open(data_dir.join("layout.blocks"), 128)?;
    let vector = (0..dimension)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 125.0)
        .collect::<Vec<_>>();
    let mut next_id = 0_usize;
    let mut blocks = 0_u64;
    while next_id < sample_nodes {
        let mut builder = FusedBlockBuilder::new(1, dimension)?;
        while next_id < sample_nodes {
            let item = HybridRecord {
                id: next_id as u64,
                assertion_time: 1,
                valid_from: 0,
                valid_to: 1_000,
                vector: vector.clone(),
                edges: (1..=edge_count)
                    .map(|step| GraphEdge {
                        target: ((next_id + step) % sample_nodes) as u64,
                        weight: 1.0,
                        edge_type: 1,
                    })
                    .collect(),
            };
            if builder.try_push(item)? {
                next_id += 1;
            } else {
                break;
            }
        }
        io.append(&builder.finish()?)?;
        blocks += 1;
    }
    io.sync()?;
    let physical_bytes = io.len();
    let bytes_per_node = physical_bytes as f64 / sample_nodes as f64;
    let extrapolated = bytes_per_node * 10_000_000.0;
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,sample_nodes,dimension,edges,blocks,physical_bytes,bytes_per_node,extrapolated_10m_bytes"
    )?;
    writeln!(
        writer,
        "engramdb,{sample_nodes},{dimension},{edge_count},{blocks},{physical_bytes},{bytes_per_node:.3},{extrapolated:.0}"
    )?;
    Ok(())
}

fn profile_fused(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let _data_dir = required_path(arguments, "--data-dir")?;
    let dimension = 128;
    let records = synthetic_records(10_000, dimension, 10);
    let mut blocks = Vec::new();
    let mut locations = std::collections::HashMap::new();
    let mut cursor = 0;
    while cursor < records.len() {
        let mut builder = FusedBlockBuilder::new(1, dimension)?;
        while cursor < records.len() {
            if builder.try_push(records[cursor].clone())? {
                cursor += 1;
            } else {
                break;
            }
        }
        let block = builder.finish()?;
        let block_index = blocks.len();
        let view = FusedBlockView::parse(block.as_slice())?;
        for slot in 0..view.len() {
            locations.insert(view.record(slot)?.id(), (block_index, slot));
        }
        blocks.push(block);
    }
    profile_fused_queries(&blocks, &locations, dimension)?;
    Ok(())
}

#[inline(never)]
fn profile_fused_queries(
    blocks: &[AlignedBlock<FUSED_BLOCK_SIZE>],
    locations: &std::collections::HashMap<u64, (usize, usize)>,
    dimension: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for query_id in 0..100 {
        let root = (query_id * 37 % 10_000) as u64;
        let mut query = vec![0_i8; dimension];
        query[root as usize % 32] = 127;
        let query_norm = 127.0_f32;
        let mut pending = std::collections::VecDeque::from([(root, 0)]);
        let mut visited = HashSet::new();
        let mut score = 0.0_f32;
        while let Some((id, depth)) = pending.pop_front() {
            if depth > 3 || !visited.insert(id) {
                continue;
            }
            let (block, slot) = locations[&id];
            let view = FusedBlockView::parse_cached(blocks[block].as_slice())?;
            let record = view.record(slot)?;
            let vector = record.quantized_vector();
            let norm = vector
                .iter()
                .map(|value| f32::from(*value).powi(2))
                .sum::<f32>()
                .sqrt();
            score += dot_i8(&query, vector) as f32 / (query_norm * norm);
            if depth < 3 {
                for edge in record.edges() {
                    pending.push_back((edge.target, depth + 1));
                }
            }
        }
        black_box(score);
    }
    Ok(())
}

fn profile_split(_arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 128;
    let records = synthetic_records(10_000, dimension, 10);
    let by_id = records
        .into_iter()
        .map(|record| (record.id, Box::new(record)))
        .collect::<std::collections::HashMap<_, _>>();
    profile_split_queries(&by_id, dimension);
    Ok(())
}

#[inline(never)]
fn profile_split_queries(
    by_id: &std::collections::HashMap<u64, Box<HybridRecord>>,
    dimension: usize,
) {
    for query_id in 0..100 {
        let root = (query_id * 37 % 10_000) as u64;
        let mut query = vec![0.0; dimension];
        query[root as usize % 32] = 1.0;
        let mut pending = std::collections::VecDeque::from([(root, 0)]);
        let mut visited = HashSet::new();
        let mut score = 0.0_f32;
        while let Some((id, depth)) = pending.pop_front() {
            if depth > 3 || !visited.insert(id) {
                continue;
            }
            let record = &by_id[&id];
            score += 1.0 - cosine_distance(&query, &record.vector);
            if depth < 3 {
                for edge in &record.edges {
                    pending.push_back((edge.target, depth + 1));
                }
            }
        }
        black_box(score);
    }
}

fn synthetic_records(nodes: usize, dimension: usize, edges: usize) -> Vec<HybridRecord> {
    (0..nodes)
        .map(|id| {
            let mut vector = vec![0.0; dimension];
            vector[id % dimension.min(32)] = 1.0;
            vector[(id * 13 + 7) % dimension] += 0.1;
            HybridRecord {
                id: id as u64,
                assertion_time: (id % 10 + 1) as u64,
                valid_from: 0,
                valid_to: 1_000,
                vector,
                edges: (1..=edges)
                    .map(|step| GraphEdge {
                        target: graph_target(id, step, nodes) as u64,
                        weight: 1.0,
                        edge_type: 1,
                    })
                    .collect(),
            }
        })
        .collect()
}

fn graph_target(id: usize, step: usize, nodes: usize) -> usize {
    let mixed = (id as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((step as u64).wrapping_mul(1_442_695_040_888_963_407));
    (mixed % nodes as u64) as usize
}

fn read_fvecs(path: &Path, limit: usize) -> io::Result<Vec<Vec<f32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut output = Vec::new();
    while output.len() < limit {
        let mut dimension = [0_u8; 4];
        match reader.read_exact(&mut dimension) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let dimension = u32::from_le_bytes(dimension) as usize;
        let mut bytes = vec![0_u8; dimension * 4];
        reader.read_exact(&mut bytes)?;
        let (values, remainder) = bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        output.push(
            values
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect(),
        );
    }
    Ok(output)
}

fn read_ivecs(path: &Path, limit: usize) -> io::Result<Vec<Vec<u32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut output = Vec::new();
    while output.len() < limit {
        let mut dimension = [0_u8; 4];
        match reader.read_exact(&mut dimension) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let dimension = u32::from_le_bytes(dimension) as usize;
        let mut bytes = vec![0_u8; dimension * 4];
        reader.read_exact(&mut bytes)?;
        let (values, remainder) = bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        output.push(
            values
                .iter()
                .map(|value| u32::from_le_bytes(*value))
                .collect(),
        );
    }
    Ok(output)
}

fn exact_l2(base: &[Vec<f32>], query: &[f32], count: usize) -> HashSet<u64> {
    let mut distances = base
        .iter()
        .enumerate()
        .map(|(id, vector)| {
            (
                id as u64,
                vector
                    .iter()
                    .zip(query)
                    .map(|(left, right)| {
                        let difference = left - right;
                        difference * difference
                    })
                    .sum::<f32>(),
            )
        })
        .collect::<Vec<_>>();
    distances.sort_by(|left, right| left.1.total_cmp(&right.1));
    distances
        .into_iter()
        .take(count)
        .map(|entry| entry.0)
        .collect()
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    1.0 - dot / (left_norm * right_norm)
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
