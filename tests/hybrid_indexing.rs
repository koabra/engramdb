use std::collections::HashSet;

use engramdb::{
    dot_i8, FusedBlockBuilder, FusedBlockView, FusedNode, GraphEdge, Hash, HybridIndex,
    TemporalPoint, TriModalQuery,
};
use tempfile::tempdir;

fn node(key: &str, vector: Vec<f32>) -> FusedNode {
    FusedNode::new(
        key,
        vec![TemporalPoint {
            assertion_time: 10,
            valid_time: 20,
        }],
        vector,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn packed_block_round_trips_without_materializing_vectors() {
    let target = Hash(*blake3::hash(b"target").as_bytes());
    let expected = FusedNode::new(
        "source",
        vec![
            TemporalPoint {
                assertion_time: 7,
                valid_time: -3,
            },
            TemporalPoint {
                assertion_time: 9,
                valid_time: 4,
            },
        ],
        vec![0.25, -0.5, 1.0, 0.0],
        vec![GraphEdge {
            target,
            weight: 0.75,
            edge_type: 3,
        }],
    )
    .unwrap();
    let expected_id = expected.id;
    let mut builder = FusedBlockBuilder::new(42);
    builder.push(expected).unwrap();
    let block = builder.finish().unwrap();

    let view = FusedBlockView::parse(block.as_slice(), 0).unwrap();
    assert_eq!(view.epoch(), 42);
    assert_eq!(view.len(), 1);
    let decoded = view.node(0, 0).unwrap();
    assert_eq!(decoded.id(), expected_id);
    assert_eq!(decoded.vector().len(), 4);
    assert_eq!(decoded.temporal().count(), 2);
    assert_eq!(decoded.edges().next().unwrap().target, target);
}

#[test]
fn fused_block_crc_rejects_corruption() {
    let mut builder = FusedBlockBuilder::new(1);
    builder.push(node("node", vec![1.0, 0.0])).unwrap();
    let mut block = builder.finish().unwrap();
    block.as_mut_slice()[4096] ^= 0x40;
    assert!(FusedBlockView::parse(block.as_slice(), 0).is_err());
}

#[test]
fn dispatched_i8_dot_matches_scalar_oracle() {
    let left: Vec<i8> = (-127_i16..=127).map(|value| value as i8).collect();
    let right: Vec<i8> = left.iter().rev().copied().collect();
    let expected: i64 = left
        .iter()
        .zip(&right)
        .map(|(left, right)| *left as i64 * *right as i64)
        .sum();
    assert_eq!(dot_i8(&left, &right), expected);
}

#[test]
fn hnsw_recall_exceeds_phase2_threshold_and_survives_reopen() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fused.dat");
    let mut index = HybridIndex::open(&path).unwrap();
    let nodes: Vec<FusedNode> = (0..320)
        .map(|item| node(&format!("item-{item}"), generated_vector(item, 64)))
        .collect();
    index.insert(1, nodes).unwrap();

    let mut recalled = 0;
    let mut expected = 0;
    for query in (0..320).step_by(7) {
        let vector = generated_vector(query, 64);
        let approximate: HashSet<Hash> = index
            .nearest(&vector, 10)
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();
        let exact = index.exact_nearest(&vector, 10).unwrap();
        expected += exact.len();
        recalled += exact
            .iter()
            .filter(|result| approximate.contains(&result.id))
            .count();
    }
    let recall = recalled as f64 / expected as f64;
    assert!(recall > 0.95, "Recall@10 was {recall:.4}");
    let stats = index.stats();
    assert_eq!(stats.nodes, 320);
    assert!(stats.logical_bytes < stats.physical_bytes);
    drop(index);

    let reopened = HybridIndex::open(path).unwrap();
    assert_eq!(reopened.stats().nodes, 320);
    assert_eq!(
        reopened.nearest(&generated_vector(17, 64), 1).unwrap()[0].id,
        Hash(*blake3::hash(b"item-17").as_bytes())
    );
}

#[test]
fn tri_modal_traversal_matches_brute_force_oracle() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("fused.dat");
    let mut nodes: Vec<FusedNode> = (0..6)
        .map(|item| {
            let vector = if matches!(item, 0 | 1 | 3 | 5) {
                vec![1.0, 0.02]
            } else {
                vec![0.0, 1.0]
            };
            node(&format!("agent-{item}"), vector)
        })
        .collect();
    let ids: Vec<Hash> = nodes.iter().map(|node| node.id).collect();
    nodes[0].edges = vec![
        GraphEdge {
            target: ids[1],
            weight: 1.0,
            edge_type: 7,
        },
        GraphEdge {
            target: ids[2],
            weight: 1.0,
            edge_type: 7,
        },
    ];
    nodes[1].edges = vec![GraphEdge {
        target: ids[3],
        weight: 1.0,
        edge_type: 7,
    }];
    nodes[2].edges = vec![GraphEdge {
        target: ids[4],
        weight: 1.0,
        edge_type: 7,
    }];
    nodes[3].edges = vec![GraphEdge {
        target: ids[5],
        weight: 1.0,
        edge_type: 8,
    }];
    nodes[3].temporal[0].assertion_time = 100;

    let mut index = HybridIndex::open(path).unwrap();
    index.insert(1, nodes).unwrap();
    let results = index
        .tri_modal_query(
            ids[0],
            TriModalQuery {
                vector: &[1.0, 0.0],
                minimum_cosine: 0.8,
                assertion_before: 50,
                valid_at: 20,
                max_hops: 3,
                edge_type: Some(7),
            },
        )
        .unwrap();
    let actual: HashSet<Hash> = results.into_iter().map(|result| result.id).collect();
    let expected = HashSet::from([ids[0], ids[1]]);
    assert_eq!(actual, expected);
}

fn generated_vector(item: usize, dimensions: usize) -> Vec<f32> {
    let cluster = item % 16;
    (0..dimensions)
        .map(|dimension| {
            let basis = if dimension % 16 == cluster { 1.0 } else { 0.0 };
            let noise = (((item * 1_103_515_245 + dimension * 12_345) >> 8) & 0xff) as f32;
            basis + (noise / 255.0 - 0.5) * 0.08
        })
        .collect()
}
