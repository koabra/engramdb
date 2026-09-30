use std::collections::{HashMap, HashSet, VecDeque};
use std::mem::size_of;

use engramdb::{
    dot_i8, DistanceMetric, FusedBlockBuilder, FusedBlockHeader, FusedBlockView, GraphEdge,
    HnswConfig, HybridIndex, HybridRecord, PackedGraphEdge, TemporalEntry, FUSED_BLOCK_SIZE,
    FUSED_HEADER_SIZE,
};
use tempfile::tempdir;

fn record(id: u64, vector: Vec<f32>, edges: Vec<GraphEdge>) -> HybridRecord {
    HybridRecord {
        id,
        assertion_time: 1,
        valid_from: 0,
        valid_to: 1_000,
        vector,
        edges,
    }
}

#[test]
fn fused_layout_is_exactly_64k_aligned_and_zero_copy_readable() {
    assert_eq!(size_of::<FusedBlockHeader>(), 64);
    assert_eq!(size_of::<TemporalEntry>(), 24);
    assert_eq!(size_of::<PackedGraphEdge>(), 16);
    assert_eq!(FUSED_HEADER_SIZE, 64);

    let mut builder = FusedBlockBuilder::new(7, 8).unwrap();
    builder
        .try_push(record(
            42,
            vec![1.0, 0.5, -0.5, 0.25, 0.0, 0.1, 0.2, 0.3],
            vec![GraphEdge {
                target: 99,
                weight: 0.75,
                edge_type: 3,
            }],
        ))
        .unwrap();
    let block = builder.finish().unwrap();
    assert!(block.is_aligned());
    assert_eq!(block.as_slice().len(), FUSED_BLOCK_SIZE);
    let view = FusedBlockView::parse(block.as_slice()).unwrap();
    assert_eq!(view.epoch(), 7);
    assert_eq!(view.dimension(), 8);
    assert_eq!(view.len(), 1);
    let stored = view.record(0).unwrap();
    assert_eq!(stored.id(), 42);
    assert_eq!(stored.quantized_vector().len(), 8);
    assert_eq!(stored.edges().collect::<Vec<_>>()[0].target, 99);
}

#[test]
fn fused_layout_detects_payload_corruption() {
    let mut builder = FusedBlockBuilder::new(1, 4).unwrap();
    builder
        .try_push(record(1, vec![1.0, 2.0, 3.0, 4.0], Vec::new()))
        .unwrap();
    let mut block = builder.finish().unwrap();
    block.as_mut_slice()[FUSED_HEADER_SIZE + 3] ^= 0xff;
    assert!(FusedBlockView::parse(block.as_slice()).is_err());
}

#[test]
fn runtime_simd_matches_scalar_dot_product() {
    let left = (-127_i16..=127)
        .map(|value| value as i8)
        .collect::<Vec<_>>();
    let right = left.iter().rev().copied().collect::<Vec<_>>();
    let expected = left
        .iter()
        .zip(&right)
        .map(|(left, right)| i64::from(*left) * i64::from(*right))
        .sum::<i64>();
    assert_eq!(dot_i8(&left, &right), expected);
}

#[test]
fn quantized_hnsw_recall_at_10_exceeds_95_percent() {
    let directory = tempdir().unwrap();
    let dimension = 64;
    let mut index = HybridIndex::open(
        directory.path(),
        dimension,
        DistanceMetric::Cosine,
        HnswConfig {
            max_connections: 24,
            ef_construction: 256,
            ef_search: 256,
        },
    )
    .unwrap();
    let mut random = 0x9e3779b97f4a7c15_u64;
    let mut vectors = Vec::new();
    let mut records = Vec::new();
    for id in 0..4_000_u64 {
        let vector = (0..dimension)
            .map(|_| {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                ((random >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32
            })
            .collect::<Vec<_>>();
        vectors.push(vector.clone());
        records.push(record(id, vector, Vec::new()));
    }
    index.insert(1, records).unwrap();

    let mut recalled = 0;
    let mut expected_total = 0;
    for query_index in (0..4_000).step_by(40) {
        let query = &vectors[query_index];
        let mut exact = vectors
            .iter()
            .enumerate()
            .map(|(id, vector)| (id as u64, cosine_distance(query, vector)))
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| left.1.total_cmp(&right.1));
        let expected = exact
            .iter()
            .take(10)
            .map(|entry| entry.0)
            .collect::<HashSet<_>>();
        let actual = index
            .nearest(query, 10, 10, u64::MAX)
            .unwrap()
            .hits
            .into_iter()
            .map(|hit| hit.id)
            .collect::<HashSet<_>>();
        recalled += expected.intersection(&actual).count();
        expected_total += expected.len();
    }
    let recall = recalled as f64 / expected_total as f64;
    assert!(recall > 0.95, "Recall@10 was {recall:.4}");
}

#[test]
fn multimodal_traversal_matches_brute_force_and_recovers() {
    let directory = tempdir().unwrap();
    let dimension = 16;
    let mut source = HashMap::new();
    let mut records = Vec::new();
    for id in 0..128_u64 {
        let medic = id % 3 == 0;
        let mut vector = vec![0.0; dimension];
        vector[if medic { 0 } else { 1 }] = 1.0;
        vector[2] = id as f32 / 1_000.0;
        let edges = (1..=3)
            .map(|step| GraphEdge {
                target: (id + step) % 128,
                weight: 1.0,
                edge_type: 1,
            })
            .collect::<Vec<_>>();
        let item = record(id, vector, edges);
        source.insert(id, item.clone());
        records.push(item);
    }
    let mut index = HybridIndex::open(
        directory.path(),
        dimension,
        DistanceMetric::Cosine,
        HnswConfig::default(),
    )
    .unwrap();
    index.insert(1, records).unwrap();
    let mut query = vec![0.0; dimension];
    query[0] = 1.0;
    let result = index.traverse(0, &query, 3, 0.8, 50, 5, |_| true).unwrap();
    let actual = result.hits.iter().map(|hit| hit.id).collect::<HashSet<_>>();
    let expected = brute_force(&source, 0, &query, 3, 0.8, 50, 5);
    assert_eq!(actual, expected);
    assert!(result.stats.physical_block_reads < result.stats.visited_records);
    drop(index);

    let reopened = HybridIndex::open(
        directory.path(),
        dimension,
        DistanceMetric::Cosine,
        HnswConfig::default(),
    )
    .unwrap();
    let recovered = reopened
        .traverse(0, &query, 3, 0.8, 50, 5, |_| true)
        .unwrap()
        .hits
        .into_iter()
        .map(|hit| hit.id)
        .collect::<HashSet<_>>();
    assert_eq!(recovered, expected);
}

fn brute_force(
    records: &HashMap<u64, HybridRecord>,
    root: u64,
    query: &[f32],
    max_hops: usize,
    threshold: f32,
    valid_at: i64,
    asserted_before: u64,
) -> HashSet<u64> {
    let mut pending = VecDeque::from([(root, 0_usize)]);
    let mut visited = HashSet::new();
    let mut hits = HashSet::new();
    while let Some((id, depth)) = pending.pop_front() {
        if depth > max_hops || !visited.insert(id) {
            continue;
        }
        let item = &records[&id];
        if item.assertion_time <= asserted_before
            && item.valid_from <= valid_at
            && valid_at < item.valid_to
        {
            if 1.0 - cosine_distance(query, &item.vector) >= threshold {
                hits.insert(id);
            }
            if depth < max_hops {
                for edge in &item.edges {
                    pending.push_back((edge.target, depth + 1));
                }
            }
        }
    }
    hits
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    1.0 - dot / (left_norm * right_norm)
}
