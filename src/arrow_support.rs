//! Arrow schemas and RecordBatch conversion for query rows and fused projections.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryBuilder, FixedSizeListArray, Float32Array, Float32Builder,
    Int64Builder, Int8Array, ListBuilder, StructBuilder, UInt16Builder, UInt32Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;

use crate::{Error, NodeProjection, QueryRow, Result};

pub fn query_rows_to_batch(rows: &[QueryRow]) -> Result<RecordBatch> {
    let mut ids = FixedSizeBinaryBuilder::with_capacity(rows.len(), 32);
    for row in rows {
        ids.append_value(row.id.0)
            .map_err(|error| Error::Arrow(error.to_string()))?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::FixedSizeBinary(32), false),
        Field::new("score", DataType::Float32, false),
        Field::new("depth", DataType::UInt32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids.finish()),
            Arc::new(Float32Array::from_iter_values(
                rows.iter().map(|row| row.score),
            )),
            Arc::new(UInt32Array::from_iter_values(
                rows.iter().map(|row| row.depth),
            )),
        ],
    )
    .map_err(|error| Error::Arrow(error.to_string()))
}

pub fn projections_to_batch(projections: &[NodeProjection]) -> Result<RecordBatch> {
    let dimensions = projections
        .first()
        .map(|projection| projection.vector.len())
        .unwrap_or(0);
    if projections
        .iter()
        .any(|projection| projection.vector.len() != dimensions)
    {
        return Err(Error::Arrow(
            "fused projections have inconsistent vector dimensions".to_owned(),
        ));
    }
    let dimensions = i32::try_from(dimensions)
        .map_err(|_| Error::Arrow("vector dimension exceeds Arrow i32".to_owned()))?;

    let mut ids = FixedSizeBinaryBuilder::with_capacity(projections.len(), 32);
    let temporal_fields = Fields::from(vec![
        Field::new("assertion_time", DataType::UInt64, false),
        Field::new("valid_time", DataType::Int64, false),
    ]);
    let temporal_values = StructBuilder::new(
        temporal_fields.clone(),
        vec![
            Box::new(UInt64Builder::new()),
            Box::new(Int64Builder::new()),
        ],
    );
    let mut temporal = ListBuilder::new(temporal_values);
    let edge_fields = Fields::from(vec![
        Field::new("target", DataType::FixedSizeBinary(32), false),
        Field::new("weight", DataType::Float32, false),
        Field::new("edge_type", DataType::UInt16, false),
    ]);
    let edge_values = StructBuilder::new(
        edge_fields.clone(),
        vec![
            Box::new(FixedSizeBinaryBuilder::new(32)),
            Box::new(Float32Builder::new()),
            Box::new(UInt16Builder::new()),
        ],
    );
    let mut edges = ListBuilder::new(edge_values);
    let mut vectors = Vec::with_capacity(projections.len() * dimensions as usize);
    let mut scales = Vec::with_capacity(projections.len());

    for projection in projections {
        ids.append_value(projection.id.0)
            .map_err(|error| Error::Arrow(error.to_string()))?;
        for point in &projection.temporal {
            temporal
                .values()
                .field_builder::<UInt64Builder>(0)
                .expect("temporal assertion builder")
                .append_value(point.assertion_time);
            temporal
                .values()
                .field_builder::<Int64Builder>(1)
                .expect("temporal valid-time builder")
                .append_value(point.valid_time);
            temporal.values().append(true);
        }
        temporal.append(true);
        vectors.extend_from_slice(&projection.vector);
        scales.push(projection.quantization_scale);
        for edge in &projection.edges {
            edges
                .values()
                .field_builder::<FixedSizeBinaryBuilder>(0)
                .expect("edge target builder")
                .append_value(edge.target.0)
                .map_err(|error| Error::Arrow(error.to_string()))?;
            edges
                .values()
                .field_builder::<Float32Builder>(1)
                .expect("edge weight builder")
                .append_value(edge.weight);
            edges
                .values()
                .field_builder::<UInt16Builder>(2)
                .expect("edge type builder")
                .append_value(edge.edge_type);
            edges.values().append(true);
        }
        edges.append(true);
    }

    let vector_values: ArrayRef = Arc::new(Int8Array::from(vectors));
    let vector_field = Arc::new(Field::new("item", DataType::Int8, false));
    let vector_array = FixedSizeListArray::try_new(vector_field, dimensions, vector_values, None)
        .map_err(|error| Error::Arrow(error.to_string()))?;
    let temporal_array = temporal.finish();
    let edge_array = edges.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::FixedSizeBinary(32), false),
        Field::new("temporal", temporal_array.data_type().clone(), false),
        Field::new("vector", vector_array.data_type().clone(), false),
        Field::new("quantization_scale", DataType::Float32, false),
        Field::new("edges", edge_array.data_type().clone(), false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids.finish()),
            Arc::new(temporal_array),
            Arc::new(vector_array),
            Arc::new(Float32Array::from(scales)),
            Arc::new(edge_array),
        ],
    )
    .map_err(|error| Error::Arrow(error.to_string()))
}
