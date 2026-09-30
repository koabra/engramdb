//! Durable speculative sessions backed by copy-on-write EngramDB branches.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::{
    execute, explain, optimize, parse_enql, plan_logical, projections_to_batch,
    query_rows_to_batches, CatalogStats, Engine, Error, FusedNode, HardwareCapabilities,
    InferenceManager, KvCacheManifest, KvCacheSnapshot, KvCacheSpec, KvRestoreTicket, Result,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Active,
    Committed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Session {
    pub id: Uuid,
    pub parent_id: Uuid,
    pub status: SessionStatus,
}

pub struct SessionManager {
    engine: Arc<Engine>,
    inference: Option<Arc<InferenceManager>>,
    sessions: RwLock<HashMap<Uuid, Session>>,
}

impl SessionManager {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            inference: None,
            sessions: RwLock::new(HashMap::new()),
        }
    }

    pub fn new_with_inference(engine: Arc<Engine>, inference: Arc<InferenceManager>) -> Self {
        Self {
            engine,
            inference: Some(inference),
            sessions: RwLock::new(HashMap::new()),
        }
    }

    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }

    pub fn fork_session(&self, parent_id: Uuid) -> Result<Session> {
        let branch = self.engine.fork(parent_id)?;
        if let Some(inference) = &self.inference {
            inference.inherit(parent_id, branch.id)?;
        }
        let session = Session {
            id: branch.id,
            parent_id,
            status: SessionStatus::Active,
        };
        self.sessions.write().insert(session.id, session);
        Ok(session)
    }

    pub fn commit_session(&self, session_id: Uuid) -> Result<Session> {
        self.engine.branch(session_id)?;
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(&session_id)
            .ok_or_else(|| Error::InvalidQuery(format!("unknown session {session_id}")))?;
        session.status = SessionStatus::Committed;
        Ok(*session)
    }

    pub fn ingest(&self, session_id: Uuid, nodes: Vec<FusedNode>) -> Result<()> {
        self.require_active(session_id)?;
        let mut transaction = self.engine.begin(session_id)?;
        for node in nodes {
            transaction.put_fused(node)?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn query(&self, session_id: Uuid, enql: &str) -> Result<RecordBatch> {
        let batches = self.query_batches(session_id, enql, 1024)?;
        if batches.len() == 1 {
            return Ok(batches.into_iter().next().unwrap());
        }
        let schema = batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| crate::query_rows_to_batch(&[]).unwrap().schema());
        arrow::compute::concat_batches(&schema, &batches)
            .map_err(|error| Error::Arrow(error.to_string()))
    }

    pub fn query_batches(
        &self,
        session_id: Uuid,
        enql: &str,
        batch_size: usize,
    ) -> Result<Vec<RecordBatch>> {
        if batch_size == 0 {
            return Err(Error::InvalidQuery(
                "execution batch size must be positive".to_owned(),
            ));
        }
        self.engine.branch(session_id)?;
        let query = parse_enql(enql)?;
        let logical = plan_logical(&query);
        let stats = CatalogStats {
            nodes: self.engine.hybrid_node_count(session_id)?.max(1),
            average_out_degree: 3.0,
        };
        let physical = optimize(logical, stats);
        let rows = execute(&self.engine, session_id, &physical)?;
        query_rows_to_batches(&rows, batch_size)
    }

    pub fn explain(&self, session_id: Uuid, enql: &str) -> Result<String> {
        self.engine.branch(session_id)?;
        let query = parse_enql(enql)?;
        let logical = plan_logical(&query);
        let physical = optimize(
            logical,
            CatalogStats {
                nodes: self.engine.hybrid_node_count(session_id)?.max(1),
                average_out_degree: 3.0,
            },
        );
        Ok(explain(&physical))
    }

    pub fn export_fused(&self, session_id: Uuid) -> Result<RecordBatch> {
        self.engine.branch(session_id)?;
        projections_to_batch(&self.engine.hybrid_projections(session_id)?)
    }

    pub fn put_kv_cache(
        &self,
        session_id: Uuid,
        spec: KvCacheSpec,
        bytes: &[u8],
    ) -> Result<KvCacheManifest> {
        self.require_active(session_id)?;
        self.inference()?.put(session_id, spec, bytes)
    }

    pub fn get_kv_cache(&self, session_id: Uuid) -> Result<Option<KvCacheSnapshot>> {
        self.engine.branch(session_id)?;
        self.inference()?.get(session_id)
    }

    pub fn kv_restore_ticket(&self, session_id: Uuid) -> Result<Option<KvRestoreTicket>> {
        self.engine.branch(session_id)?;
        self.inference()?.restore_ticket(session_id)
    }

    pub fn hardware_capabilities(&self) -> Result<&HardwareCapabilities> {
        Ok(self.inference()?.capabilities())
    }

    pub fn session(&self, session_id: Uuid) -> Option<Session> {
        self.sessions.read().get(&session_id).copied()
    }

    fn require_active(&self, session_id: Uuid) -> Result<()> {
        let sessions = self.sessions.read();
        let session = sessions
            .get(&session_id)
            .ok_or_else(|| Error::InvalidQuery(format!("unknown session {session_id}")))?;
        if session.status != SessionStatus::Active {
            return Err(Error::InvalidQuery(format!(
                "session {session_id} is already committed"
            )));
        }
        Ok(())
    }

    fn inference(&self) -> Result<&Arc<InferenceManager>> {
        self.inference.as_ref().ok_or_else(|| {
            Error::HardwareUnavailable("server was started without KV-cache storage".to_owned())
        })
    }
}
