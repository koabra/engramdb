//! EnQL parsing, logical planning, cost-based access selection, and execution.

use uuid::Uuid;

use crate::{Engine, Error, Hash, Result, TriModalQuery};

#[derive(Debug, Clone, PartialEq)]
pub enum EnqlQuery {
    Nearest {
        vector: Vec<f32>,
        limit: usize,
    },
    Traverse {
        start: Hash,
        vector: Vec<f32>,
        minimum_cosine: f32,
        assertion_before: u64,
        valid_at: i64,
        max_hops: usize,
        edge_type: Option<u16>,
        as_of_system_time: Option<u64>,
        limit: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalPlan {
    pub source: LogicalSource,
    pub temporal_filter: Option<TemporalFilter>,
    pub projection: Vec<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalSource {
    VectorScan {
        vector: Vec<f32>,
    },
    Traverse {
        start: Hash,
        vector: Vec<f32>,
        minimum_cosine: f32,
        max_hops: usize,
        edge_type: Option<u16>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalFilter {
    pub assertion_before: u64,
    pub valid_at: i64,
    pub as_of_system_time: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPath {
    VectorOnly,
    VectorFirst,
    GraphFirst,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalPlan {
    pub logical: LogicalPlan,
    pub access_path: AccessPath,
    pub estimated_cost: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct CatalogStats {
    pub nodes: usize,
    pub average_out_degree: f64,
}

impl Default for CatalogStats {
    fn default() -> Self {
        Self {
            nodes: 1,
            average_out_degree: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryRow {
    pub id: Hash,
    pub score: f32,
    pub depth: u32,
}

pub fn parse_enql(input: &str) -> Result<EnqlQuery> {
    let query = input.trim();
    let upper = query.to_ascii_uppercase();
    if upper.starts_with("VECTOR NEAREST") {
        let vector = parse_vector_after(query, &upper, "VECTOR NEAREST")?;
        let limit = parse_usize_after(query, &upper, "LIMIT")?;
        return Ok(EnqlQuery::Nearest { vector, limit });
    }
    if !upper.starts_with("MATCH") {
        return Err(Error::InvalidQuery(
            "query must begin with MATCH or VECTOR NEAREST".to_owned(),
        ));
    }
    let start = parse_hash_after(query, &upper, "FROM")?;
    let max_hops = if let Some(position) = upper.find("HOPS") {
        parse_usize_at(query, position + "HOPS".len())?
    } else {
        parse_cypher_hops(query)?
    };
    let vector = parse_vector_after(query, &upper, "VECTOR")?;
    let minimum_cosine = parse_f32_after(query, &upper, "COSINE >")?;
    let assertion_before = parse_u64_after(query, &upper, "ASSERTION <")?;
    let valid_at = parse_i64_after(query, &upper, "VALID <=")?;
    let edge_type = upper
        .find("EDGE_TYPE =")
        .map(|position| parse_u16_at(query, position + "EDGE_TYPE =".len()))
        .transpose()?;
    let as_of_system_time = upper
        .find("AS OF SYSTEM TIME")
        .map(|position| parse_u64_at(query, position + "AS OF SYSTEM TIME".len()))
        .transpose()?;
    let limit = parse_usize_after(query, &upper, "LIMIT")?;
    if !(0.0..=1.0).contains(&minimum_cosine) || max_hops == 0 || limit == 0 {
        return Err(Error::InvalidQuery(
            "cosine must be in [0,1], and hops/limit must be positive".to_owned(),
        ));
    }
    Ok(EnqlQuery::Traverse {
        start,
        vector,
        minimum_cosine,
        assertion_before,
        valid_at,
        max_hops,
        edge_type,
        as_of_system_time,
        limit,
    })
}

pub fn plan_logical(query: &EnqlQuery) -> LogicalPlan {
    match query {
        EnqlQuery::Nearest { vector, limit } => LogicalPlan {
            source: LogicalSource::VectorScan {
                vector: vector.clone(),
            },
            temporal_filter: None,
            projection: vec!["id".to_owned(), "score".to_owned()],
            limit: *limit,
        },
        EnqlQuery::Traverse {
            start,
            vector,
            minimum_cosine,
            assertion_before,
            valid_at,
            max_hops,
            edge_type,
            as_of_system_time,
            limit,
        } => LogicalPlan {
            source: LogicalSource::Traverse {
                start: *start,
                vector: vector.clone(),
                minimum_cosine: *minimum_cosine,
                max_hops: *max_hops,
                edge_type: *edge_type,
            },
            temporal_filter: Some(TemporalFilter {
                assertion_before: *assertion_before,
                valid_at: *valid_at,
                as_of_system_time: *as_of_system_time,
            }),
            projection: vec!["id".to_owned(), "score".to_owned(), "depth".to_owned()],
            limit: *limit,
        },
    }
}

pub fn optimize(logical: LogicalPlan, stats: CatalogStats) -> PhysicalPlan {
    match &logical.source {
        LogicalSource::VectorScan { .. } => PhysicalPlan {
            logical,
            access_path: AccessPath::VectorOnly,
            estimated_cost: (stats.nodes.max(1) as f64).log2(),
        },
        LogicalSource::Traverse {
            minimum_cosine,
            max_hops,
            edge_type,
            ..
        } => {
            let vector_selectivity = (1.0 - *minimum_cosine as f64).clamp(0.0001, 1.0);
            let vector_cost = stats.nodes as f64 * vector_selectivity;
            let graph_cost = stats
                .average_out_degree
                .max(1.0)
                .powi((*max_hops).min(i32::MAX as usize) as i32);
            let access_path = if *minimum_cosine > 0.98 {
                AccessPath::VectorFirst
            } else if edge_type.is_some() || *max_hops <= 2 || graph_cost <= vector_cost {
                AccessPath::GraphFirst
            } else {
                AccessPath::VectorFirst
            };
            PhysicalPlan {
                logical,
                access_path,
                estimated_cost: match access_path {
                    AccessPath::VectorFirst => vector_cost,
                    AccessPath::GraphFirst => graph_cost,
                    AccessPath::VectorOnly => unreachable!(),
                },
            }
        }
    }
}

pub fn explain(plan: &PhysicalPlan) -> String {
    format!(
        "PhysicalPlan(access={:?}, estimated_cost={:.3}, limit={}, projection=[{}])",
        plan.access_path,
        plan.estimated_cost,
        plan.logical.limit,
        plan.logical.projection.join(",")
    )
}

pub fn execute(engine: &Engine, branch: Uuid, plan: &PhysicalPlan) -> Result<Vec<QueryRow>> {
    match &plan.logical.source {
        LogicalSource::VectorScan { vector } => Ok(engine
            .hybrid_nearest(branch, vector, plan.logical.limit)?
            .into_iter()
            .map(|result| QueryRow {
                id: result.id,
                score: result.score,
                depth: 0,
            })
            .collect()),
        LogicalSource::Traverse {
            start,
            vector,
            minimum_cosine,
            max_hops,
            edge_type,
        } => {
            let temporal = plan.logical.temporal_filter.ok_or_else(|| {
                Error::Invariant("traversal plan is missing temporal filter".to_owned())
            })?;
            let assertion_before = temporal
                .as_of_system_time
                .map(|time| temporal.assertion_before.min(time.saturating_add(1)))
                .unwrap_or(temporal.assertion_before);
            let query = TriModalQuery {
                vector,
                minimum_cosine: *minimum_cosine,
                assertion_before,
                valid_at: temporal.valid_at,
                max_hops: *max_hops,
                edge_type: *edge_type,
            };
            let matches = match plan.access_path {
                AccessPath::VectorFirst => {
                    engine.hybrid_query_vector_first(branch, *start, query, plan.logical.limit)?
                }
                AccessPath::GraphFirst => engine.hybrid_query(branch, *start, query)?,
                AccessPath::VectorOnly => unreachable!(),
            };
            Ok(matches
                .into_iter()
                .take(plan.logical.limit)
                .map(|result| QueryRow {
                    id: result.id,
                    score: result.cosine_similarity,
                    depth: result.depth as u32,
                })
                .collect())
        }
    }
}

fn parse_vector_after(input: &str, upper: &str, keyword: &str) -> Result<Vec<f32>> {
    let position = upper
        .find(keyword)
        .ok_or_else(|| Error::InvalidQuery(format!("missing {keyword}")))?;
    let suffix = &input[position + keyword.len()..];
    let start = suffix
        .find('[')
        .ok_or_else(|| Error::InvalidQuery("vector is missing `[`".to_owned()))?;
    let end = suffix[start + 1..]
        .find(']')
        .map(|end| start + 1 + end)
        .ok_or_else(|| Error::InvalidQuery("vector is missing `]`".to_owned()))?;
    let vector = suffix[start + 1..end]
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f32>()
                .map_err(|_| Error::InvalidQuery("vector contains a non-f32 value".to_owned()))
        })
        .collect::<Result<Vec<_>>>()?;
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidVector);
    }
    Ok(vector)
}

fn parse_hash_after(input: &str, upper: &str, keyword: &str) -> Result<Hash> {
    let position = upper
        .find(keyword)
        .ok_or_else(|| Error::InvalidQuery(format!("missing {keyword}")))?;
    let token = next_token(input, position + keyword.len())?;
    if token.len() != 64 {
        return Err(Error::InvalidQuery(
            "node hash must contain 64 hexadecimal characters".to_owned(),
        ));
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in token.as_bytes().chunks_exact(2).enumerate() {
        let value = std::str::from_utf8(pair)
            .ok()
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .ok_or_else(|| Error::InvalidQuery("node hash is not hexadecimal".to_owned()))?;
        bytes[index] = value;
    }
    Ok(Hash(bytes))
}

fn parse_cypher_hops(input: &str) -> Result<usize> {
    let marker = input
        .find('*')
        .ok_or_else(|| Error::InvalidQuery("missing HOPS or Cypher edge range".to_owned()))?;
    let suffix = &input[marker + 1..];
    let range_end = suffix
        .find(']')
        .ok_or_else(|| Error::InvalidQuery("unterminated Cypher edge range".to_owned()))?;
    let range = &suffix[..range_end];
    let maximum = range.split("..").last().unwrap_or(range).trim();
    maximum
        .parse()
        .map_err(|_| Error::InvalidQuery("invalid Cypher hop range".to_owned()))
}

fn parse_usize_after(input: &str, upper: &str, keyword: &str) -> Result<usize> {
    let position = upper
        .rfind(keyword)
        .ok_or_else(|| Error::InvalidQuery(format!("missing {keyword}")))?;
    parse_usize_at(input, position + keyword.len())
}

fn parse_u64_after(input: &str, upper: &str, keyword: &str) -> Result<u64> {
    let position = upper
        .find(keyword)
        .ok_or_else(|| Error::InvalidQuery(format!("missing {keyword}")))?;
    parse_u64_at(input, position + keyword.len())
}

fn parse_i64_after(input: &str, upper: &str, keyword: &str) -> Result<i64> {
    let position = upper
        .find(keyword)
        .ok_or_else(|| Error::InvalidQuery(format!("missing {keyword}")))?;
    next_token(input, position + keyword.len())?
        .parse()
        .map_err(|_| Error::InvalidQuery(format!("invalid integer after {keyword}")))
}

fn parse_f32_after(input: &str, upper: &str, keyword: &str) -> Result<f32> {
    let position = upper
        .find(keyword)
        .ok_or_else(|| Error::InvalidQuery(format!("missing {keyword}")))?;
    next_token(input, position + keyword.len())?
        .parse()
        .map_err(|_| Error::InvalidQuery(format!("invalid number after {keyword}")))
}

fn parse_usize_at(input: &str, position: usize) -> Result<usize> {
    next_token(input, position)?
        .parse()
        .map_err(|_| Error::InvalidQuery("invalid positive integer".to_owned()))
}

fn parse_u64_at(input: &str, position: usize) -> Result<u64> {
    next_token(input, position)?
        .parse()
        .map_err(|_| Error::InvalidQuery("invalid unsigned integer".to_owned()))
}

fn parse_u16_at(input: &str, position: usize) -> Result<u16> {
    next_token(input, position)?
        .parse()
        .map_err(|_| Error::InvalidQuery("invalid edge type".to_owned()))
}

fn next_token(input: &str, position: usize) -> Result<&str> {
    input[position..]
        .trim_start()
        .split_whitespace()
        .next()
        .map(|token| token.trim_matches(|character: char| ",;)".contains(character)))
        .filter(|token| !token.is_empty())
        .ok_or_else(|| Error::InvalidQuery("missing query value".to_owned()))
}
