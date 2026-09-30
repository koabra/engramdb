use std::sync::Arc;

use arrow::array::{FixedSizeListArray, ListArray};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::Ticket;
use futures::{StreamExt, TryStreamExt};

use engramdb::{
    explain, optimize, parse_enql, plan_logical, AccessPath, CatalogStats, Engine,
    EngramFlightService, EnqlQuery, FusedNode, GraphEdge, Hash, SessionManager, SessionStatus,
    TemporalPoint,
};
use tempfile::tempdir;
use tonic::Request;

fn node(key: &str, vector: [f32; 2], edges: Vec<GraphEdge>) -> FusedNode {
    FusedNode::new(
        key,
        vec![TemporalPoint {
            assertion_time: 10,
            valid_time: 20,
        }],
        vector.to_vec(),
        edges,
    )
    .unwrap()
}

fn traversal_query(start: Hash, threshold: f32, edge_type: Option<u16>) -> String {
    format!(
        "MATCH (a)-[:EDGE*1..3]->(b) FROM {} VECTOR [1.0,0.0] \
         COSINE > {threshold} ASSERTION < 100 VALID <= 20 {} \
         AS OF SYSTEM TIME 99 RETURN id,score,depth LIMIT 10",
        start,
        edge_type
            .map(|value| format!("EDGE_TYPE = {value}"))
            .unwrap_or_default()
    )
}

#[test]
fn enql_parser_and_optimizer_choose_restrictive_access_path() {
    let start = Hash(*blake3::hash(b"start").as_bytes());
    let selective = parse_enql(&traversal_query(start, 0.99, None)).unwrap();
    let selective_plan = optimize(
        plan_logical(&selective),
        CatalogStats {
            nodes: 1_000_000,
            average_out_degree: 8.0,
        },
    );
    assert_eq!(selective_plan.access_path, AccessPath::VectorFirst);
    assert!(explain(&selective_plan).contains("VectorFirst"));

    let strict_graph = parse_enql(&traversal_query(start, 0.8, Some(7))).unwrap();
    let graph_plan = optimize(
        plan_logical(&strict_graph),
        CatalogStats {
            nodes: 1_000_000,
            average_out_degree: 8.0,
        },
    );
    assert_eq!(graph_plan.access_path, AccessPath::GraphFirst);

    assert!(matches!(
        parse_enql("VECTOR NEAREST [1.0, 0.0] LIMIT 5").unwrap(),
        EnqlQuery::Nearest { limit: 5, .. }
    ));
}

#[test]
fn vector_and_graph_first_plans_return_the_same_rows() {
    let directory = tempdir().unwrap();
    let engine = Arc::new(Engine::open(directory.path()).unwrap());
    let sessions = SessionManager::new(Arc::clone(&engine));
    let main = engine.main_branch().id;
    let session = sessions.fork_session(main).unwrap();
    let alpha = Hash(*blake3::hash(b"alpha").as_bytes());
    let beta = Hash(*blake3::hash(b"beta").as_bytes());
    sessions
        .ingest(
            session.id,
            vec![
                node(
                    "alpha",
                    [1.0, 0.0],
                    vec![GraphEdge {
                        target: beta,
                        weight: 1.0,
                        edge_type: 7,
                    }],
                ),
                node("beta", [0.999, 0.001], Vec::new()),
            ],
        )
        .unwrap();

    let graph = sessions
        .query(session.id, &traversal_query(alpha, 0.8, Some(7)))
        .unwrap();
    let vector = sessions
        .query(session.id, &traversal_query(alpha, 0.99, None))
        .unwrap();
    assert_eq!(graph.num_rows(), 2);
    assert_eq!(vector.num_rows(), 2);

    let nearest = sessions
        .query(session.id, "VECTOR NEAREST [1.0,0.0] LIMIT 1")
        .unwrap();
    assert_eq!(nearest.num_rows(), 1);
}

#[test]
fn fused_projection_maps_to_nested_arrow_arrays() {
    let directory = tempdir().unwrap();
    let engine = Arc::new(Engine::open(directory.path()).unwrap());
    let sessions = SessionManager::new(Arc::clone(&engine));
    let session = sessions.fork_session(engine.main_branch().id).unwrap();
    sessions
        .ingest(session.id, vec![node("alpha", [1.0, 0.0], Vec::new())])
        .unwrap();
    let batch = sessions.export_fused(session.id).unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert!(batch
        .column_by_name("vector")
        .unwrap()
        .as_any()
        .is::<FixedSizeListArray>());
    assert!(batch
        .column_by_name("temporal")
        .unwrap()
        .as_any()
        .is::<ListArray>());
    assert!(batch
        .column_by_name("edges")
        .unwrap()
        .as_any()
        .is::<ListArray>());
}

#[test]
fn committed_sessions_reject_further_ingestion() {
    let directory = tempdir().unwrap();
    let engine = Arc::new(Engine::open(directory.path()).unwrap());
    let sessions = SessionManager::new(Arc::clone(&engine));
    let session = sessions.fork_session(engine.main_branch().id).unwrap();
    let committed = sessions.commit_session(session.id).unwrap();
    assert_eq!(committed.status, SessionStatus::Committed);
    assert!(sessions
        .ingest(session.id, vec![node("late", [1.0, 0.0], Vec::new())])
        .is_err());
}

#[tokio::test]
async fn flight_actions_and_query_stream_return_arrow_batches() {
    let directory = tempdir().unwrap();
    let engine = Arc::new(Engine::open(directory.path()).unwrap());
    let manager = Arc::new(SessionManager::new(Arc::clone(&engine)));
    let session = manager.fork_session(engine.main_branch().id).unwrap();
    manager
        .ingest(session.id, vec![node("alpha", [1.0, 0.0], Vec::new())])
        .unwrap();
    let service = EngramFlightService::new(manager);
    let ticket = Ticket {
        ticket: format!("{}\nVECTOR NEAREST [1.0,0.0] LIMIT 1", session.id).into(),
    };
    let flight_data = service
        .do_get(Request::new(ticket))
        .await
        .unwrap()
        .into_inner()
        .map_err(FlightError::from);
    let mut batches = FlightRecordBatchStream::new_from_flight_data(flight_data);
    let batch = batches.next().await.unwrap().unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert!(batches.next().await.is_none());
}
