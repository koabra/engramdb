use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use engramdb::{
    dot_i8, DirectIo, FusedBlockBuilder, FusedBlockView, FusedNode, GraphEdge, Hash, HybridIndex,
    QuantizedVector, TemporalPoint, TriModalQuery, VectorMetric, FUSED_BLOCK_LAYOUT,
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
        "recall" => recall(&arguments[1..]),
        "query" => query(&arguments[1..]),
        "amplification" => amplification(&arguments[1..]),
        "cache-layout" => cache_layout(&arguments[1..]),
        _ => {
            eprintln!(
                "usage:\n  phase2-bench recall --data-file PATH --output CSV \
                 [--nodes 10000] [--queries 200] [--dimensions 128]\n  \
                 phase2-bench query --data-file PATH --output CSV \
                 [--nodes 2000] [--queries 10000] [--dimensions 128]\n  \
                 phase2-bench amplification --data-file PATH --output CSV \
                 [--nodes 10000000] [--dimensions 768] [--edges 10]\n  \
                 phase2-bench cache-layout --output CSV [--iterations 10000]"
            );
            Ok(())
        }
    }
}

fn recall(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = required_path(arguments, "--data-file")?;
    let output = required_path(arguments, "--output")?;
    if let Some(base_path) = option(arguments, "--base") {
        return sift_recall(arguments, &path, &output, Path::new(&base_path));
    }
    let nodes = usize_option(arguments, "--nodes", 10_000)?;
    let queries = usize_option(arguments, "--queries", 200)?;
    let dimensions = usize_option(arguments, "--dimensions", 128)?;
    reset_file(&path)?;
    let mut index = HybridIndex::open(&path)?;
    index.insert(
        1,
        (0..nodes)
            .map(|item| generated_node(item, dimensions, 0))
            .collect::<Result<Vec<_>, _>>()?,
    )?;

    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "dataset,oracle,query,k,intersection,recall,approximate_us,exact_us,nodes,dimensions,ef_search"
    )?;
    for query_id in 0..queries {
        let vector = generated_vector(query_id * 37 % nodes, dimensions);
        let started = Instant::now();
        let approximate = index.nearest(&vector, 10)?;
        let approximate_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        let started = Instant::now();
        let exact = index.exact_nearest(&vector, 10)?;
        let exact_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        let intersection = exact
            .iter()
            .filter(|expected| approximate.iter().any(|actual| actual.id == expected.id))
            .count();
        writeln!(
            writer,
            "synthetic-clustered,exact-quantized-scan,{query_id},10,{intersection},{:.6},{approximate_us:.3},{exact_us:.3},{nodes},{dimensions},256",
            intersection as f64 / 10.0
        )?;
    }
    Ok(())
}

fn sift_recall(
    arguments: &[String],
    data_path: &Path,
    output: &Path,
    base_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_path = required_path(arguments, "--queries-file")?;
    let groundtruth_path = required_path(arguments, "--groundtruth")?;
    let limit_nodes = option(arguments, "--limit-nodes")
        .map(|value| value.parse())
        .transpose()?;
    let limit_queries = usize_option(arguments, "--limit-queries", 200)?;
    let ef_search = usize_option(arguments, "--ef-search", 4096)?;
    let base = read_fvecs(base_path, limit_nodes)?;
    let queries = read_fvecs(&query_path, Some(limit_queries))?;
    let groundtruth = read_ivecs(&groundtruth_path, Some(queries.len()))?;
    if base.is_empty() || queries.is_empty() || base[0].len() != queries[0].len() {
        return Err("SIFT base/query files are empty or dimensionally inconsistent".into());
    }
    if groundtruth.len() != queries.len() || groundtruth.iter().any(|row| row.len() < 10) {
        return Err("SIFT ground truth does not contain ten neighbors per query".into());
    }
    if groundtruth
        .iter()
        .flat_map(|row| row.iter().take(10))
        .any(|index| *index >= base.len())
    {
        return Err(
            "official SIFT ground truth references vectors excluded by --limit-nodes".into(),
        );
    }

    reset_file(data_path)?;
    let nodes = base.len();
    let dimensions = base[0].len();
    let mut index = HybridIndex::open_with_metric(data_path, VectorMetric::SquaredL2)?;
    index.insert(
        1,
        base.into_iter()
            .enumerate()
            .map(|(item, vector)| {
                FusedNode::new(
                    format!("sift-{item}"),
                    vec![TemporalPoint {
                        assertion_time: 1,
                        valid_time: 1,
                    }],
                    vector,
                    Vec::new(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let ordinal_by_id: std::collections::HashMap<Hash, usize> = (0..nodes)
        .map(|item| {
            (
                Hash(*blake3::hash(format!("sift-{item}").as_bytes()).as_bytes()),
                item,
            )
        })
        .collect();

    let mut writer = output_writer(output)?;
    writeln!(
        writer,
        "dataset,oracle,query,k,intersection,recall,approximate_us,exact_us,nodes,dimensions,ef_search"
    )?;
    for (query_id, vector) in queries.iter().enumerate() {
        let started = Instant::now();
        let approximate = index.nearest_with_ef(vector, 10, ef_search)?;
        let approximate_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        let actual: std::collections::HashSet<usize> = approximate
            .iter()
            .map(|result| ordinal_by_id[&result.id])
            .collect();
        let intersection = groundtruth[query_id]
            .iter()
            .take(10)
            .filter(|expected| actual.contains(expected))
            .count();
        writeln!(
            writer,
            "SIFT1M,official-fp32-groundtruth,{query_id},10,{intersection},{:.6},{approximate_us:.3},0.000,{nodes},{dimensions},{ef_search}",
            intersection as f64 / 10.0
        )?;
    }
    Ok(())
}

fn query(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = required_path(arguments, "--data-file")?;
    let output = required_path(arguments, "--output")?;
    let nodes = usize_option(arguments, "--nodes", 2_000)?;
    let queries = usize_option(arguments, "--queries", 10_000)?;
    let dimensions = usize_option(arguments, "--dimensions", 128)?;
    reset_file(&path)?;

    let ids: Vec<Hash> = (0..nodes)
        .map(|item| Hash(*blake3::hash(format!("node-{item}").as_bytes()).as_bytes()))
        .collect();
    let mut records = Vec::with_capacity(nodes);
    for item in 0..nodes {
        let edges = (1..=3)
            .map(|step| GraphEdge {
                target: ids[(item + step) % nodes],
                weight: 1.0 / step as f32,
                edge_type: 1,
            })
            .collect();
        records.push(FusedNode::new(
            format!("node-{item}"),
            vec![TemporalPoint {
                assertion_time: (item % 100) as u64,
                valid_time: (item % 500) as i64,
            }],
            generated_vector(item, dimensions),
            edges,
        )?);
    }
    let mut index = HybridIndex::open(&path)?;
    index.insert(1, records)?;

    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,query,latency_us,results,nodes,dimensions,hops,minimum_cosine"
    )?;
    for query_id in 0..queries {
        let start = query_id * 7919 % nodes;
        let vector = generated_vector(start, dimensions);
        let started = Instant::now();
        let results = index.tri_modal_query(
            ids[start],
            TriModalQuery {
                vector: &vector,
                minimum_cosine: 0.8,
                assertion_before: 75,
                valid_at: 400,
                max_hops: 3,
                edge_type: Some(1),
            },
        )?;
        let latency_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        writeln!(
            writer,
            "engramdb,{query_id},{latency_us:.3},{},{nodes},{dimensions},3,0.8",
            results.len()
        )?;
    }
    Ok(())
}

fn amplification(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = required_path(arguments, "--data-file")?;
    let output = required_path(arguments, "--output")?;
    let nodes = usize_option(arguments, "--nodes", 10_000_000)?;
    let dimensions = usize_option(arguments, "--dimensions", 768)?;
    let edge_count = usize_option(arguments, "--edges", 10)?;
    reset_file(&path)?;
    let io = DirectIo::open_with_layout(&path, 256, FUSED_BLOCK_LAYOUT)?;
    let vector: Vec<f32> = (0..dimensions)
        .map(|dimension| ((dimension * 31 % 251) as f32 - 125.0) / 125.0)
        .collect();
    let edges: Vec<GraphEdge> = (0..edge_count)
        .map(|edge| GraphEdge {
            target: Hash(*blake3::hash(format!("target-{edge}").as_bytes()).as_bytes()),
            weight: 1.0,
            edge_type: 1,
        })
        .collect();
    let mut builder = FusedBlockBuilder::new(1);
    let mut blocks = 0_u64;
    let started = Instant::now();
    for item in 0..nodes {
        let node = FusedNode::new(
            format!("amp-{item}"),
            vec![TemporalPoint {
                assertion_time: 1,
                valid_time: 1,
            }],
            vector.clone(),
            edges.clone(),
        )?;
        if !builder.is_empty() && !builder.can_fit(&node)? {
            io.append(&builder.finish()?)?;
            blocks += 1;
            builder = FusedBlockBuilder::new(1);
        }
        builder.push(node)?;
    }
    if !builder.is_empty() {
        io.append(&builder.finish()?)?;
        blocks += 1;
    }
    io.sync()?;
    let elapsed = started.elapsed().as_secs_f64();
    let physical_bytes = fs::metadata(&path)?.len();
    let source_bytes_per_node = 32 + dimensions * 4 + 16 + edge_count * 40;
    let encoded_bytes_per_node = 64 + dimensions + 16 + edge_count * 40;
    let source_bytes = nodes as u64 * source_bytes_per_node as u64;
    let encoded_bytes = nodes as u64 * encoded_bytes_per_node as u64;
    let mut writer = output_writer(&output)?;
    writeln!(
        writer,
        "product,nodes,dimensions,edges,blocks,source_bytes,encoded_bytes,physical_bytes,physical_vs_source,packing_amplification,elapsed_s,measured"
    )?;
    writeln!(
        writer,
        "engramdb,{nodes},{dimensions},{edge_count},{blocks},{source_bytes},{encoded_bytes},{physical_bytes},{:.6},{:.6},{elapsed:.6},true",
        physical_bytes as f64 / source_bytes as f64,
        physical_bytes as f64 / encoded_bytes as f64
    )?;
    Ok(())
}

fn cache_layout(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let output = required_path(arguments, "--output")?;
    let iterations = usize_option(arguments, "--iterations", 10_000)?;
    let dimensions = 256;
    let nodes = 128;
    let records = (0..nodes)
        .map(|item| generated_node(item, dimensions, 2))
        .collect::<Result<Vec<_>, _>>()?;
    let mut builder = FusedBlockBuilder::new(1);
    for node in records {
        builder.push(node)?;
    }
    let block = builder.finish()?;
    let view = FusedBlockView::parse(block.as_slice(), 0)?;
    let query = QuantizedVector::from_f32(&generated_vector(7, dimensions))?;
    let pointer_vectors: Vec<Box<[i8]>> = (0..view.len())
        .map(|slot| {
            view.node(slot, 0)
                .unwrap()
                .vector()
                .to_vec()
                .into_boxed_slice()
        })
        .collect();

    let started = Instant::now();
    let mut fused_sum = 0_i64;
    for _ in 0..iterations {
        for slot in 0..view.len() {
            fused_sum += dot_i8(view.node(slot, 0)?.vector(), query.codes());
        }
    }
    let fused_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let mut pointer_sum = 0_i64;
    for _ in 0..iterations {
        for vector in &pointer_vectors {
            pointer_sum += dot_i8(vector, query.codes());
        }
    }
    let pointer_ns = started.elapsed().as_nanos();
    std::hint::black_box((fused_sum, pointer_sum));

    let mut writer = output_writer(&output)?;
    writeln!(writer, "layout,iterations,nodes,dimensions,elapsed_ns")?;
    writeln!(writer, "fused,{iterations},{nodes},{dimensions},{fused_ns}")?;
    writeln!(
        writer,
        "pointer,{iterations},{nodes},{dimensions},{pointer_ns}"
    )?;
    Ok(())
}

fn generated_node(
    item: usize,
    dimensions: usize,
    edges: usize,
) -> Result<FusedNode, engramdb::Error> {
    FusedNode::new(
        format!("node-{item}"),
        vec![TemporalPoint {
            assertion_time: (item % 100) as u64,
            valid_time: (item % 500) as i64,
        }],
        generated_vector(item, dimensions),
        (0..edges)
            .map(|edge| GraphEdge {
                target: Hash(
                    *blake3::hash(format!("node-{}", (item + edge + 1) % 10_000).as_bytes())
                        .as_bytes(),
                ),
                weight: 1.0,
                edge_type: 1,
            })
            .collect(),
    )
}

fn generated_vector(item: usize, dimensions: usize) -> Vec<f32> {
    let cluster = item % 32;
    (0..dimensions)
        .map(|dimension| {
            let basis = if dimension % 32 == cluster { 1.0 } else { 0.0 };
            let noise = (((item * 1_103_515_245 + dimension * 12_345) >> 8) & 0xff) as f32;
            basis + (noise / 255.0 - 0.5) * 0.08
        })
        .collect()
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

fn reset_file(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn output_writer(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    File::create(path)
}

fn read_fvecs(
    path: &Path,
    limit: Option<usize>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut output = Vec::new();
    loop {
        if limit.is_some_and(|limit| output.len() >= limit) {
            break;
        }
        let mut dimension_bytes = [0_u8; 4];
        match reader.read_exact(&mut dimension_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        let dimension = u32::from_le_bytes(dimension_bytes) as usize;
        if dimension == 0 || dimension > u16::MAX as usize {
            return Err(format!("invalid fvec dimension {dimension}").into());
        }
        let mut bytes = vec![0_u8; dimension * 4];
        reader.read_exact(&mut bytes)?;
        output.push(
            bytes
                .chunks_exact(4)
                .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                .collect(),
        );
    }
    Ok(output)
}

fn read_ivecs(
    path: &Path,
    limit: Option<usize>,
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut output = Vec::new();
    loop {
        if limit.is_some_and(|limit| output.len() >= limit) {
            break;
        }
        let mut dimension_bytes = [0_u8; 4];
        match reader.read_exact(&mut dimension_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        let dimension = u32::from_le_bytes(dimension_bytes) as usize;
        let mut bytes = vec![0_u8; dimension * 4];
        reader.read_exact(&mut bytes)?;
        output.push(
            bytes
                .chunks_exact(4)
                .map(|value| u32::from_le_bytes(value.try_into().unwrap()) as usize)
                .collect(),
        );
    }
    Ok(output)
}
