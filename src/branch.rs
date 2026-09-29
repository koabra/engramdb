//! Durable branch, transaction, recovery, and temporal merge APIs.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use uuid::Uuid;

use crate::io::{DirectIo, IoStats};
use crate::tree::{Hash, NodeRef, NodeStore, PersistentTree};
use crate::{Error, Result};

const META_MAGIC: &[u8; 8] = b"ENGMETA1";
const META_HEADER: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalRecord {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub valid_from: i64,
    pub valid_to: i64,
    pub asserted_at: u64,
}

impl TemporalRecord {
    pub fn new(
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
        valid_from: i64,
        valid_to: i64,
    ) -> Result<Self> {
        if valid_from >= valid_to {
            return Err(Error::InvalidInterval);
        }
        Ok(Self {
            key: key.into(),
            value: value.into(),
            valid_from,
            valid_to,
            asserted_at: 0,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub root_hash: Hash,
    pub epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultPoint {
    None,
    AfterPageWrites,
    AfterDataSync,
    DuringMetadataAppend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    pub root_hash: Hash,
    pub applied_ranges: usize,
    pub epoch: u64,
}

#[derive(Clone)]
struct BranchState {
    branch: Branch,
    root: NodeRef,
    fork_root: NodeRef,
}

struct EngineState {
    branches: HashMap<Uuid, BranchState>,
    main: Uuid,
    epoch: u64,
}

pub struct Engine {
    directory: PathBuf,
    store: Arc<NodeStore>,
    tree: PersistentTree,
    metadata: MetadataLog,
    state: RwLock<EngineState>,
}

impl Engine {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory)?;
        let io = Arc::new(DirectIo::open(directory.join("pages.dat"), 256)?);
        let store = Arc::new(NodeStore::open(io, 4096)?);
        let tree = PersistentTree::new(Arc::clone(&store));
        let metadata = MetadataLog::open(directory.join("branches.log"))?;
        let events = metadata.replay()?;

        let mut branches = HashMap::new();
        let mut main = None;
        let mut epoch = 0;
        for event in events {
            match event {
                MetadataEvent::Create {
                    id,
                    parent,
                    root,
                    fork_root,
                    epoch: event_epoch,
                } => {
                    validate_root(&tree, &store, root)?;
                    if let Some(parent_id) = parent {
                        if !branches.contains_key(&parent_id) {
                            return Err(Error::CorruptMetadata {
                                offset: 0,
                                reason: format!("branch {id} has unknown parent {parent_id}"),
                            });
                        }
                    } else if main.replace(id).is_some() {
                        return Err(Error::CorruptMetadata {
                            offset: 0,
                            reason: "multiple root branches".to_owned(),
                        });
                    }
                    epoch = epoch.max(event_epoch);
                    branches.insert(
                        id,
                        BranchState {
                            branch: Branch {
                                id,
                                parent_id: parent,
                                root_hash: root.hash,
                                epoch: event_epoch,
                            },
                            root,
                            fork_root,
                        },
                    );
                }
                MetadataEvent::Commit {
                    id,
                    root,
                    epoch: event_epoch,
                } => {
                    validate_root(&tree, &store, root)?;
                    let state = branches.get_mut(&id).ok_or(Error::UnknownBranch(id))?;
                    state.root = root;
                    state.branch.root_hash = root.hash;
                    state.branch.epoch = event_epoch;
                    epoch = epoch.max(event_epoch);
                }
                MetadataEvent::Merge {
                    target,
                    source,
                    root,
                    epoch: event_epoch,
                } => {
                    validate_root(&tree, &store, root)?;
                    if !branches.contains_key(&source) {
                        return Err(Error::UnknownBranch(source));
                    }
                    let state = branches
                        .get_mut(&target)
                        .ok_or(Error::UnknownBranch(target))?;
                    state.root = root;
                    state.branch.root_hash = root.hash;
                    state.branch.epoch = event_epoch;
                    epoch = epoch.max(event_epoch);
                }
            }
        }

        if branches.is_empty() {
            let root = tree.empty_root()?;
            store.sync()?;
            let id = Uuid::new_v4();
            let event = MetadataEvent::Create {
                id,
                parent: None,
                root,
                fork_root: root,
                epoch: 0,
            };
            metadata.append(&event, false)?;
            main = Some(id);
            branches.insert(
                id,
                BranchState {
                    branch: Branch {
                        id,
                        parent_id: None,
                        root_hash: root.hash,
                        epoch: 0,
                    },
                    root,
                    fork_root: root,
                },
            );
        }

        Ok(Self {
            directory,
            store,
            tree,
            metadata,
            state: RwLock::new(EngineState {
                branches,
                main: main.expect("non-empty metadata has a root branch"),
                epoch,
            }),
        })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn main_branch(&self) -> Branch {
        let state = self.state.read();
        state.branches[&state.main].branch.clone()
    }

    pub fn branch(&self, id: Uuid) -> Result<Branch> {
        self.state
            .read()
            .branches
            .get(&id)
            .map(|state| state.branch.clone())
            .ok_or(Error::UnknownBranch(id))
    }

    /// Durably creates a branch by writing only metadata; no tree pages are
    /// copied, so work and storage are independent of database size.
    pub fn fork(&self, parent: Uuid) -> Result<Branch> {
        let mut state = self.state.write();
        let parent_state = state
            .branches
            .get(&parent)
            .cloned()
            .ok_or(Error::UnknownBranch(parent))?;
        state.epoch += 1;
        let branch = Branch {
            id: Uuid::new_v4(),
            parent_id: Some(parent),
            root_hash: parent_state.root.hash,
            epoch: state.epoch,
        };
        self.metadata.append(
            &MetadataEvent::Create {
                id: branch.id,
                parent: Some(parent),
                root: parent_state.root,
                fork_root: parent_state.root,
                epoch: branch.epoch,
            },
            false,
        )?;
        state.branches.insert(
            branch.id,
            BranchState {
                branch: branch.clone(),
                root: parent_state.root,
                fork_root: parent_state.root,
            },
        );
        Ok(branch)
    }

    pub fn begin(&self, branch: Uuid) -> Result<Transaction<'_>> {
        let root = self
            .state
            .read()
            .branches
            .get(&branch)
            .map(|state| state.root)
            .ok_or(Error::UnknownBranch(branch))?;
        Ok(Transaction {
            engine: self,
            branch,
            expected_root: root,
            writes: Vec::new(),
        })
    }

    pub fn get(&self, branch: Uuid, key: &[u8], valid_at: i64) -> Result<Option<TemporalRecord>> {
        self.get_as_of(branch, key, valid_at, u64::MAX)
    }

    pub fn get_as_of(
        &self,
        branch: Uuid,
        key: &[u8],
        valid_at: i64,
        asserted_at: u64,
    ) -> Result<Option<TemporalRecord>> {
        let root = self
            .state
            .read()
            .branches
            .get(&branch)
            .map(|state| state.root)
            .ok_or(Error::UnknownBranch(branch))?;
        let mut best = None;
        for (tree_key, value) in self.tree.entries(root)? {
            let record = decode_record(&tree_key, value)?;
            if record.key == key
                && record.valid_from <= valid_at
                && valid_at < record.valid_to
                && record.asserted_at <= asserted_at
                && best
                    .as_ref()
                    .map(|current: &TemporalRecord| current.asserted_at < record.asserted_at)
                    .unwrap_or(true)
            {
                best = Some(record);
            }
        }
        Ok(best)
    }

    pub fn merge(&self, target: Uuid, source: Uuid) -> Result<MergeOutcome> {
        let mut state = self.state.write();
        let target_state = state
            .branches
            .get(&target)
            .cloned()
            .ok_or(Error::UnknownBranch(target))?;
        let source_state = state
            .branches
            .get(&source)
            .cloned()
            .ok_or(Error::UnknownBranch(source))?;
        let base_root = merge_base(&target_state, &source_state)?;

        let base = logical_ranges(&self.tree, base_root)?;
        let target_ranges = logical_ranges(&self.tree, target_state.root)?;
        let source_ranges = logical_ranges(&self.tree, source_state.root)?;
        let target_changes = changes(&base, &target_ranges);
        let source_changes = changes(&base, &source_ranges);

        let conflicts = source_changes
            .iter()
            .filter(|(source_key, source_value)| {
                target_changes.iter().any(|(target_key, target_value)| {
                    source_key.0 == target_key.0
                        && overlaps(source_key.1, source_key.2, target_key.1, target_key.2)
                        && source_value != &target_value
                })
            })
            .count();
        if conflicts > 0 {
            return Err(Error::MergeConflict(conflicts));
        }

        state.epoch += 1;
        let merge_epoch = state.epoch;
        let mut root = target_state.root;
        for ((key, valid_from, valid_to), value) in &source_changes {
            let record = TemporalRecord {
                key: key.clone(),
                value: value.clone(),
                valid_from: *valid_from,
                valid_to: *valid_to,
                asserted_at: merge_epoch,
            };
            root = self
                .tree
                .insert(root, encode_record_key(&record), record.value.clone())?;
        }
        self.store.sync()?;
        self.metadata.append(
            &MetadataEvent::Merge {
                target,
                source,
                root,
                epoch: merge_epoch,
            },
            false,
        )?;
        let target_mut = state.branches.get_mut(&target).unwrap();
        target_mut.root = root;
        target_mut.branch.root_hash = root.hash;
        target_mut.branch.epoch = merge_epoch;
        Ok(MergeOutcome {
            root_hash: root.hash,
            applied_ranges: source_changes.len(),
            epoch: merge_epoch,
        })
    }

    pub fn validate(&self) -> Result<()> {
        for branch in self.state.read().branches.values() {
            validate_root(&self.tree, &self.store, branch.root)?;
        }
        Ok(())
    }

    pub fn io_stats(&self) -> IoStats {
        self.store.io().stats()
    }

    pub fn page_count(&self) -> usize {
        self.store.page_count()
    }
}

pub struct Transaction<'a> {
    engine: &'a Engine,
    branch: Uuid,
    expected_root: NodeRef,
    writes: Vec<TemporalRecord>,
}

impl Transaction<'_> {
    pub fn put(&mut self, record: TemporalRecord) -> Result<()> {
        if record.valid_from >= record.valid_to {
            return Err(Error::InvalidInterval);
        }
        self.writes.push(record);
        Ok(())
    }

    pub fn commit(self) -> Result<Branch> {
        self.commit_with_fault(FaultPoint::None)
    }

    pub fn commit_with_fault(self, fault: FaultPoint) -> Result<Branch> {
        let mut state = self.engine.state.write();
        let current = state
            .branches
            .get(&self.branch)
            .cloned()
            .ok_or(Error::UnknownBranch(self.branch))?;
        if current.root != self.expected_root {
            return Err(Error::StaleTransaction(self.branch));
        }
        state.epoch += 1;
        let commit_epoch = state.epoch;
        let mut root = current.root;
        for mut record in self.writes {
            record.asserted_at = commit_epoch;
            root =
                self.engine
                    .tree
                    .insert(root, encode_record_key(&record), record.value.clone())?;
        }
        if fault == FaultPoint::AfterPageWrites {
            return Err(Error::InjectedFault("after page writes"));
        }
        self.engine.store.sync()?;
        if fault == FaultPoint::AfterDataSync {
            return Err(Error::InjectedFault("after data sync"));
        }
        self.engine.metadata.append(
            &MetadataEvent::Commit {
                id: self.branch,
                root,
                epoch: commit_epoch,
            },
            fault == FaultPoint::DuringMetadataAppend,
        )?;
        if fault == FaultPoint::DuringMetadataAppend {
            return Err(Error::InjectedFault("during metadata append"));
        }
        let branch_state = state.branches.get_mut(&self.branch).unwrap();
        branch_state.root = root;
        branch_state.branch.root_hash = root.hash;
        branch_state.branch.epoch = commit_epoch;
        Ok(branch_state.branch.clone())
    }
}

fn validate_root(tree: &PersistentTree, store: &NodeStore, root: NodeRef) -> Result<()> {
    if !store.contains(root) {
        return Err(Error::CorruptPage {
            offset: root.offset,
            reason: format!("committed root {} is missing", root.hash),
        });
    }
    tree.entries(root)?;
    Ok(())
}

type RangeKey = (Vec<u8>, i64, i64);

fn logical_ranges(tree: &PersistentTree, root: NodeRef) -> Result<BTreeMap<RangeKey, Vec<u8>>> {
    let mut latest: BTreeMap<RangeKey, (u64, Vec<u8>)> = BTreeMap::new();
    for (key, value) in tree.entries(root)? {
        let record = decode_record(&key, value)?;
        let range_key = (record.key, record.valid_from, record.valid_to);
        let entry = latest
            .entry(range_key)
            .or_insert((record.asserted_at, record.value.clone()));
        if record.asserted_at >= entry.0 {
            *entry = (record.asserted_at, record.value);
        }
    }
    Ok(latest
        .into_iter()
        .map(|(key, (_, value))| (key, value))
        .collect())
}

fn changes(
    base: &BTreeMap<RangeKey, Vec<u8>>,
    branch: &BTreeMap<RangeKey, Vec<u8>>,
) -> BTreeMap<RangeKey, Vec<u8>> {
    branch
        .iter()
        .filter(|(key, value)| base.get(*key) != Some(*value))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn merge_base(target: &BranchState, source: &BranchState) -> Result<NodeRef> {
    if source.branch.parent_id == Some(target.branch.id) {
        Ok(source.fork_root)
    } else if target.branch.parent_id == Some(source.branch.id) {
        Ok(target.fork_root)
    } else if source.branch.parent_id == target.branch.parent_id
        && source.fork_root == target.fork_root
    {
        Ok(source.fork_root)
    } else {
        Err(Error::Invariant(
            "Phase 1 merge supports parent/child and sibling branches from one root".to_owned(),
        ))
    }
}

fn overlaps(left_from: i64, left_to: i64, right_from: i64, right_to: i64) -> bool {
    left_from < right_to && right_from < left_to
}

fn encode_record_key(record: &TemporalRecord) -> Vec<u8> {
    let mut output = Vec::with_capacity(4 + record.key.len() + 24);
    output.extend_from_slice(&(record.key.len() as u32).to_be_bytes());
    output.extend_from_slice(&record.key);
    output.extend_from_slice(&ordered_i64(record.valid_from));
    output.extend_from_slice(&ordered_i64(record.valid_to));
    output.extend_from_slice(&record.asserted_at.to_be_bytes());
    output
}

fn decode_record(key: &[u8], value: Vec<u8>) -> Result<TemporalRecord> {
    if key.len() < 4 {
        return Err(Error::Invariant("temporal key is truncated".to_owned()));
    }
    let key_len = u32::from_be_bytes(key[..4].try_into().unwrap()) as usize;
    if key.len() != 4 + key_len + 24 {
        return Err(Error::Invariant(
            "temporal key has invalid length".to_owned(),
        ));
    }
    let cursor = 4 + key_len;
    Ok(TemporalRecord {
        key: key[4..cursor].to_vec(),
        value,
        valid_from: decode_ordered_i64(key[cursor..cursor + 8].try_into().unwrap()),
        valid_to: decode_ordered_i64(key[cursor + 8..cursor + 16].try_into().unwrap()),
        asserted_at: u64::from_be_bytes(key[cursor + 16..cursor + 24].try_into().unwrap()),
    })
}

fn ordered_i64(value: i64) -> [u8; 8] {
    ((value as u64) ^ (1 << 63)).to_be_bytes()
}

fn decode_ordered_i64(bytes: [u8; 8]) -> i64 {
    (u64::from_be_bytes(bytes) ^ (1 << 63)) as i64
}

#[derive(Debug)]
enum MetadataEvent {
    Create {
        id: Uuid,
        parent: Option<Uuid>,
        root: NodeRef,
        fork_root: NodeRef,
        epoch: u64,
    },
    Commit {
        id: Uuid,
        root: NodeRef,
        epoch: u64,
    },
    Merge {
        target: Uuid,
        source: Uuid,
        root: NodeRef,
        epoch: u64,
    },
}

struct MetadataLog {
    path: PathBuf,
    writer: Mutex<File>,
}

impl MetadataLog {
    fn open(path: PathBuf) -> Result<Self> {
        let writer = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(writer),
        })
    }

    fn replay(&self) -> Result<Vec<MetadataEvent>> {
        let mut file = File::open(&self.path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut position = 0;
        let mut events = Vec::new();
        while position < bytes.len() {
            if bytes.len() - position < META_HEADER {
                break;
            }
            if &bytes[position..position + 8] != META_MAGIC {
                return Err(Error::CorruptMetadata {
                    offset: position as u64,
                    reason: "invalid record magic".to_owned(),
                });
            }
            let length =
                u32::from_le_bytes(bytes[position + 8..position + 12].try_into().unwrap()) as usize;
            let expected_crc =
                u32::from_le_bytes(bytes[position + 12..position + 16].try_into().unwrap());
            let end = position
                .checked_add(META_HEADER)
                .and_then(|header| header.checked_add(length))
                .ok_or_else(|| Error::CorruptMetadata {
                    offset: position as u64,
                    reason: "record length overflow".to_owned(),
                })?;
            if end > bytes.len() {
                break;
            }
            let payload = &bytes[position + META_HEADER..end];
            if crc32fast::hash(payload) != expected_crc {
                return Err(Error::CorruptMetadata {
                    offset: position as u64,
                    reason: "record CRC32 mismatch".to_owned(),
                });
            }
            events.push(
                decode_event(payload).map_err(|error| Error::CorruptMetadata {
                    offset: position as u64,
                    reason: error.to_string(),
                })?,
            );
            position = end;
        }
        if position != bytes.len() {
            let writer = self.writer.lock();
            writer.set_len(position as u64)?;
            writer.sync_data()?;
        }
        Ok(events)
    }

    fn append(&self, event: &MetadataEvent, partial: bool) -> Result<()> {
        let payload = encode_event(event);
        let mut record = Vec::with_capacity(META_HEADER + payload.len());
        record.extend_from_slice(META_MAGIC);
        record.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        record.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        record.extend_from_slice(&payload);
        let mut writer = self.writer.lock();
        writer.seek(SeekFrom::End(0))?;
        if partial {
            writer.write_all(&record[..record.len() / 2])?;
            writer.sync_data()?;
            return Ok(());
        }
        writer.write_all(&record)?;
        writer.sync_data()?;
        Ok(())
    }
}

fn encode_event(event: &MetadataEvent) -> Vec<u8> {
    let mut output = Vec::new();
    match event {
        MetadataEvent::Create {
            id,
            parent,
            root,
            fork_root,
            epoch,
        } => {
            output.push(1);
            output.extend_from_slice(id.as_bytes());
            output.push(parent.is_some() as u8);
            if let Some(parent) = parent {
                output.extend_from_slice(parent.as_bytes());
            }
            put_ref(&mut output, *root);
            put_ref(&mut output, *fork_root);
            output.extend_from_slice(&epoch.to_le_bytes());
        }
        MetadataEvent::Commit { id, root, epoch } => {
            output.push(2);
            output.extend_from_slice(id.as_bytes());
            put_ref(&mut output, *root);
            output.extend_from_slice(&epoch.to_le_bytes());
        }
        MetadataEvent::Merge {
            target,
            source,
            root,
            epoch,
        } => {
            output.push(3);
            output.extend_from_slice(target.as_bytes());
            output.extend_from_slice(source.as_bytes());
            put_ref(&mut output, *root);
            output.extend_from_slice(&epoch.to_le_bytes());
        }
    }
    output
}

fn decode_event(payload: &[u8]) -> Result<MetadataEvent> {
    let mut cursor = MetaCursor {
        bytes: payload,
        position: 0,
    };
    let tag = cursor.take::<1>()?[0];
    let event = match tag {
        1 => {
            let id = Uuid::from_bytes(cursor.take::<16>()?);
            let parent = match cursor.take::<1>()?[0] {
                0 => None,
                1 => Some(Uuid::from_bytes(cursor.take::<16>()?)),
                _ => return Err(Error::Invariant("invalid parent marker".to_owned())),
            };
            MetadataEvent::Create {
                id,
                parent,
                root: cursor.reference()?,
                fork_root: cursor.reference()?,
                epoch: u64::from_le_bytes(cursor.take::<8>()?),
            }
        }
        2 => MetadataEvent::Commit {
            id: Uuid::from_bytes(cursor.take::<16>()?),
            root: cursor.reference()?,
            epoch: u64::from_le_bytes(cursor.take::<8>()?),
        },
        3 => MetadataEvent::Merge {
            target: Uuid::from_bytes(cursor.take::<16>()?),
            source: Uuid::from_bytes(cursor.take::<16>()?),
            root: cursor.reference()?,
            epoch: u64::from_le_bytes(cursor.take::<8>()?),
        },
        _ => return Err(Error::Invariant("unknown metadata event".to_owned())),
    };
    if cursor.position != payload.len() {
        return Err(Error::Invariant(
            "trailing bytes in metadata event".to_owned(),
        ));
    }
    Ok(event)
}

fn put_ref(output: &mut Vec<u8>, reference: NodeRef) {
    output.extend_from_slice(&reference.hash.0);
    output.extend_from_slice(&reference.offset.to_le_bytes());
}

struct MetaCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl MetaCursor<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.position + N > self.bytes.len() {
            return Err(Error::Invariant("metadata event is truncated".to_owned()));
        }
        let value = self.bytes[self.position..self.position + N]
            .try_into()
            .unwrap();
        self.position += N;
        Ok(value)
    }

    fn reference(&mut self) -> Result<NodeRef> {
        Ok(NodeRef {
            hash: Hash(self.take::<32>()?),
            offset: u64::from_le_bytes(self.take::<8>()?),
        })
    }
}
