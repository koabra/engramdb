//! Immutable content-addressed B+ tree nodes.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::buffer_pool::BufferPool;
use crate::io::{AlignedPage, DirectIo, PAGE_SIZE};
use crate::{Error, Result};

const PAGE_MAGIC: &[u8; 8] = b"ENGRAM01";
const PAGE_VERSION: u8 = 1;
const HEADER_SIZE: usize = 64;
const MAX_PAYLOAD: usize = PAGE_SIZE - HEADER_SIZE;
const LEAF_TAG: u8 = 1;
const INTERNAL_TAG: u8 = 2;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeRef {
    pub hash: Hash,
    pub offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Leaf(Vec<(Vec<u8>, Vec<u8>)>),
    Internal {
        separators: Vec<Vec<u8>>,
        children: Vec<NodeRef>,
    },
}

pub(crate) struct NodeStore {
    io: Arc<DirectIo>,
    pool: BufferPool,
    index: RwLock<HashMap<Hash, u64>>,
}

impl NodeStore {
    pub fn open(io: Arc<DirectIo>, cache_pages: usize) -> Result<Self> {
        let pool = BufferPool::new(Arc::clone(&io), cache_pages);
        let store = Self {
            io,
            pool,
            index: RwLock::new(HashMap::new()),
        };
        store.rebuild_index()?;
        Ok(store)
    }

    fn rebuild_index(&self) -> Result<()> {
        let mut index = self.index.write();
        let mut offset = 0;
        while offset < self.io.len() {
            let page = self.io.read(offset)?;
            match decode_page(page.as_slice(), offset) {
                Ok((hash, _)) => {
                    index.entry(hash).or_insert(offset);
                    offset += PAGE_SIZE as u64;
                }
                Err(_error) if offset + PAGE_SIZE as u64 == self.io.len() => {
                    // The append-only file can end in a torn, uncommitted page.
                    // Metadata replay later proves whether any committed root
                    // referenced data beyond this point.
                    self.io.truncate(offset)?;
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn put(&self, node: &Node) -> Result<NodeRef> {
        let payload = encode_node(node)?;
        let hash = Hash(*blake3::hash(&payload).as_bytes());
        let mut index = self.index.write();
        if let Some(offset) = index.get(&hash).copied() {
            return Ok(NodeRef { hash, offset });
        }
        let page = encode_page(hash, &payload)?;
        let offset = self.io.append(&page)?;
        index.insert(hash, offset);
        Ok(NodeRef { hash, offset })
    }

    fn get(&self, reference: NodeRef) -> Result<Node> {
        if self.index.read().get(&reference.hash).copied() != Some(reference.offset) {
            return Err(Error::CorruptPage {
                offset: reference.offset,
                reason: format!("hash {} is absent from the page index", reference.hash),
            });
        }
        let page = self.pool.get(reference.offset)?;
        let (actual_hash, node) = decode_page(page.bytes(), reference.offset)?;
        if actual_hash != reference.hash {
            return Err(Error::CorruptPage {
                offset: reference.offset,
                reason: format!(
                    "parent expected hash {}, page contains {}",
                    reference.hash, actual_hash
                ),
            });
        }
        Ok(node)
    }

    pub(crate) fn contains(&self, reference: NodeRef) -> bool {
        self.index.read().get(&reference.hash).copied() == Some(reference.offset)
            && self.get(reference).is_ok()
    }

    pub(crate) fn sync(&self) -> Result<()> {
        self.io.sync()
    }

    pub(crate) fn page_count(&self) -> usize {
        self.index.read().len()
    }

    pub(crate) fn io(&self) -> &Arc<DirectIo> {
        &self.io
    }
}

pub(crate) struct PersistentTree {
    store: Arc<NodeStore>,
}

struct Inserted {
    left: NodeRef,
    split: Option<(Vec<u8>, NodeRef)>,
}

impl PersistentTree {
    pub fn new(store: Arc<NodeStore>) -> Self {
        Self { store }
    }

    pub fn empty_root(&self) -> Result<NodeRef> {
        self.store.put(&Node::Leaf(Vec::new()))
    }

    pub fn insert(&self, root: NodeRef, key: Vec<u8>, value: Vec<u8>) -> Result<NodeRef> {
        let inserted = self.insert_at(root, key, value)?;
        if let Some((separator, right)) = inserted.split {
            self.store.put(&Node::Internal {
                separators: vec![separator],
                children: vec![inserted.left, right],
            })
        } else {
            Ok(inserted.left)
        }
    }

    fn insert_at(&self, reference: NodeRef, key: Vec<u8>, value: Vec<u8>) -> Result<Inserted> {
        match self.store.get(reference)? {
            Node::Leaf(mut entries) => {
                match entries.binary_search_by(|(candidate, _)| candidate.cmp(&key)) {
                    Ok(index) => entries[index].1 = value,
                    Err(index) => entries.insert(index, (key, value)),
                }
                if encode_node(&Node::Leaf(entries.clone())).is_ok() {
                    return Ok(Inserted {
                        left: self.store.put(&Node::Leaf(entries))?,
                        split: None,
                    });
                }
                if entries.len() < 2 {
                    return Err(Error::NodeTooLarge {
                        actual: encode_leaf_size(&entries),
                        maximum: MAX_PAYLOAD,
                    });
                }
                let split_index = (1..entries.len())
                    .filter(|index| {
                        encode_node(&Node::Leaf(entries[..*index].to_vec())).is_ok()
                            && encode_node(&Node::Leaf(entries[*index..].to_vec())).is_ok()
                    })
                    .min_by_key(|index| {
                        let left = encode_leaf_size(&entries[..*index]);
                        let right = encode_leaf_size(&entries[*index..]);
                        left.abs_diff(right)
                    })
                    .ok_or_else(|| Error::NodeTooLarge {
                        actual: encode_leaf_size(&entries),
                        maximum: MAX_PAYLOAD,
                    })?;
                let right_entries = entries.split_off(split_index);
                let separator = right_entries[0].0.clone();
                let left = self.store.put(&Node::Leaf(entries))?;
                let right = self.store.put(&Node::Leaf(right_entries))?;
                Ok(Inserted {
                    left,
                    split: Some((separator, right)),
                })
            }
            Node::Internal {
                mut separators,
                mut children,
            } => {
                let child_index =
                    separators.partition_point(|separator| key.as_slice() >= separator);
                let inserted = self.insert_at(children[child_index], key, value)?;
                children[child_index] = inserted.left;
                if let Some((separator, right)) = inserted.split {
                    separators.insert(child_index, separator);
                    children.insert(child_index + 1, right);
                }
                let candidate = Node::Internal {
                    separators: separators.clone(),
                    children: children.clone(),
                };
                if encode_node(&candidate).is_ok() {
                    return Ok(Inserted {
                        left: self.store.put(&candidate)?,
                        split: None,
                    });
                }
                if children.len() < 3 {
                    return Err(Error::NodeTooLarge {
                        actual: encode_internal_size(&separators, &children),
                        maximum: MAX_PAYLOAD,
                    });
                }
                let midpoint = (1..children.len())
                    .filter(|midpoint| {
                        *midpoint >= 2
                            && children.len() - *midpoint >= 2
                            && encode_node(&Node::Internal {
                                separators: separators[..*midpoint - 1].to_vec(),
                                children: children[..*midpoint].to_vec(),
                            })
                            .is_ok()
                            && encode_node(&Node::Internal {
                                separators: separators[*midpoint..].to_vec(),
                                children: children[*midpoint..].to_vec(),
                            })
                            .is_ok()
                    })
                    .min_by_key(|midpoint| {
                        let left = encode_internal_size(
                            &separators[..*midpoint - 1],
                            &children[..*midpoint],
                        );
                        let right =
                            encode_internal_size(&separators[*midpoint..], &children[*midpoint..]);
                        left.abs_diff(right)
                    })
                    .ok_or_else(|| Error::NodeTooLarge {
                        actual: encode_internal_size(&separators, &children),
                        maximum: MAX_PAYLOAD,
                    })?;
                let promoted = separators[midpoint - 1].clone();
                let right_children = children.split_off(midpoint);
                let right_separators = separators.split_off(midpoint);
                separators.pop();
                let left = self.store.put(&Node::Internal {
                    separators,
                    children,
                })?;
                let right = self.store.put(&Node::Internal {
                    separators: right_separators,
                    children: right_children,
                })?;
                Ok(Inserted {
                    left,
                    split: Some((promoted, right)),
                })
            }
        }
    }

    pub fn entries(&self, root: NodeRef) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut output = Vec::new();
        self.collect(root, &mut output)?;
        Ok(output)
    }

    fn collect(&self, reference: NodeRef, output: &mut Vec<(Vec<u8>, Vec<u8>)>) -> Result<()> {
        match self.store.get(reference)? {
            Node::Leaf(entries) => output.extend(entries),
            Node::Internal { children, .. } => {
                for child in children {
                    self.collect(child, output)?;
                }
            }
        }
        Ok(())
    }
}

fn encode_page(hash: Hash, payload: &[u8]) -> Result<AlignedPage> {
    if payload.len() > MAX_PAYLOAD {
        return Err(Error::NodeTooLarge {
            actual: payload.len(),
            maximum: MAX_PAYLOAD,
        });
    }
    let mut page = AlignedPage::zeroed();
    let bytes = page.as_mut_slice();
    bytes[..8].copy_from_slice(PAGE_MAGIC);
    bytes[8] = PAGE_VERSION;
    bytes[12..16].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes[16..48].copy_from_slice(&hash.0);
    bytes[48..52].copy_from_slice(&crc32fast::hash(payload).to_le_bytes());
    bytes[HEADER_SIZE..HEADER_SIZE + payload.len()].copy_from_slice(payload);
    Ok(page)
}

fn decode_page(bytes: &[u8], offset: u64) -> Result<(Hash, Node)> {
    if bytes.len() != PAGE_SIZE || &bytes[..8] != PAGE_MAGIC || bytes[8] != PAGE_VERSION {
        return Err(Error::CorruptPage {
            offset,
            reason: "invalid magic, version, or page size".to_owned(),
        });
    }
    let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    if payload_len > MAX_PAYLOAD {
        return Err(Error::CorruptPage {
            offset,
            reason: "payload length exceeds page".to_owned(),
        });
    }
    let hash = Hash(bytes[16..48].try_into().unwrap());
    let expected_crc = u32::from_le_bytes(bytes[48..52].try_into().unwrap());
    let payload = &bytes[HEADER_SIZE..HEADER_SIZE + payload_len];
    if crc32fast::hash(payload) != expected_crc {
        return Err(Error::CorruptPage {
            offset,
            reason: "CRC32 mismatch".to_owned(),
        });
    }
    if *blake3::hash(payload).as_bytes() != hash.0 {
        return Err(Error::CorruptPage {
            offset,
            reason: "BLAKE3 mismatch".to_owned(),
        });
    }
    Ok((hash, decode_node(payload, offset)?))
}

fn encode_node(node: &Node) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    match node {
        Node::Leaf(entries) => {
            output.push(LEAF_TAG);
            put_u32(&mut output, entries.len() as u32);
            for (key, value) in entries {
                put_bytes(&mut output, key);
                put_bytes(&mut output, value);
            }
        }
        Node::Internal {
            separators,
            children,
        } => {
            if children.len() != separators.len() + 1 {
                return Err(Error::Invariant(
                    "internal node must have one more child than separator".to_owned(),
                ));
            }
            output.push(INTERNAL_TAG);
            put_u32(&mut output, children.len() as u32);
            for child in children {
                output.extend_from_slice(&child.hash.0);
                output.extend_from_slice(&child.offset.to_le_bytes());
            }
            put_u32(&mut output, separators.len() as u32);
            for separator in separators {
                put_bytes(&mut output, separator);
            }
        }
    }
    if output.len() > MAX_PAYLOAD {
        return Err(Error::NodeTooLarge {
            actual: output.len(),
            maximum: MAX_PAYLOAD,
        });
    }
    Ok(output)
}

fn decode_node(payload: &[u8], offset: u64) -> Result<Node> {
    let mut cursor = Cursor::new(payload, offset);
    let tag = cursor.byte()?;
    let node = match tag {
        LEAF_TAG => {
            let count = cursor.u32()? as usize;
            let mut entries = Vec::with_capacity(count);
            for _ in 0..count {
                entries.push((cursor.bytes()?, cursor.bytes()?));
            }
            if !entries.windows(2).all(|pair| pair[0].0 < pair[1].0) {
                return Err(cursor.corrupt("leaf keys are not strictly ordered"));
            }
            Node::Leaf(entries)
        }
        INTERNAL_TAG => {
            let child_count = cursor.u32()? as usize;
            if child_count == 0 {
                return Err(cursor.corrupt("internal node has no children"));
            }
            let mut children = Vec::with_capacity(child_count);
            for _ in 0..child_count {
                let hash = Hash(cursor.fixed::<32>()?);
                let child_offset = u64::from_le_bytes(cursor.fixed::<8>()?);
                children.push(NodeRef {
                    hash,
                    offset: child_offset,
                });
            }
            let separator_count = cursor.u32()? as usize;
            if separator_count + 1 != child_count {
                return Err(cursor.corrupt("invalid internal node cardinality"));
            }
            let mut separators = Vec::with_capacity(separator_count);
            for _ in 0..separator_count {
                separators.push(cursor.bytes()?);
            }
            if !separators.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(cursor.corrupt("separators are not strictly ordered"));
            }
            Node::Internal {
                separators,
                children,
            }
        }
        _ => return Err(cursor.corrupt("unknown node type")),
    };
    if cursor.position != payload.len() {
        return Err(cursor.corrupt("trailing bytes in node payload"));
    }
    Ok(node)
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(output: &mut Vec<u8>, value: &[u8]) {
    put_u32(output, value.len() as u32);
    output.extend_from_slice(value);
}

fn encode_leaf_size(entries: &[(Vec<u8>, Vec<u8>)]) -> usize {
    1 + 4
        + entries
            .iter()
            .map(|(key, value)| 8 + key.len() + value.len())
            .sum::<usize>()
}

fn encode_internal_size(separators: &[Vec<u8>], children: &[NodeRef]) -> usize {
    1 + 4
        + children.len() * 40
        + 4
        + separators
            .iter()
            .map(|separator| 4 + separator.len())
            .sum::<usize>()
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
    offset: u64,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: u64) -> Self {
        Self {
            bytes,
            position: 0,
            offset,
        }
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.fixed::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed::<4>()?))
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.position + N > self.bytes.len() {
            return Err(self.corrupt("node payload is truncated"));
        }
        let value = self.bytes[self.position..self.position + N]
            .try_into()
            .unwrap();
        self.position += N;
        Ok(value)
    }

    fn bytes(&mut self) -> Result<Vec<u8>> {
        let length = self.u32()? as usize;
        if self.position + length > self.bytes.len() {
            return Err(self.corrupt("length-prefixed field is truncated"));
        }
        let value = self.bytes[self.position..self.position + length].to_vec();
        self.position += length;
        Ok(value)
    }

    fn corrupt(&self, reason: &str) -> Error {
        Error::CorruptPage {
            offset: self.offset,
            reason: reason.to_owned(),
        }
    }
}
