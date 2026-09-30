//! Arrow Flight transport for sessions, queries, and fused-node ingestion.

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, Int64Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, Result as ActionResult, SchemaResult,
    Ticket,
};
use futures::{stream, Stream, StreamExt, TryStreamExt};
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

use crate::{FusedNode, SessionManager, TemporalPoint};

type ResponseStream<T> =
    Pin<Box<dyn Stream<Item = std::result::Result<T, Status>> + Send + 'static>>;

pub struct EngramFlightService {
    sessions: Arc<SessionManager>,
}

impl EngramFlightService {
    pub fn new(sessions: Arc<SessionManager>) -> Self {
        Self { sessions }
    }
}

#[tonic::async_trait]
impl FlightService for EngramFlightService {
    type HandshakeStream = ResponseStream<HandshakeResponse>;
    type ListFlightsStream = ResponseStream<FlightInfo>;
    type DoGetStream = ResponseStream<FlightData>;
    type DoPutStream = ResponseStream<PutResult>;
    type DoExchangeStream = ResponseStream<FlightData>;
    type DoActionStream = ResponseStream<ActionResult>;
    type ListActionsStream = ResponseStream<ActionType>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> std::result::Result<Response<Self::HandshakeStream>, Status> {
        Ok(Response::new(Box::pin(stream::empty())))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> std::result::Result<Response<Self::ListFlightsStream>, Status> {
        Ok(Response::new(Box::pin(stream::empty())))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("use DoGet with an EnQL ticket"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("polling is not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> std::result::Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented(
            "query schema is returned in the DoGet stream",
        ))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> std::result::Result<Response<Self::DoGetStream>, Status> {
        let (session, query) = parse_session_payload(&request.into_inner().ticket)?;
        let batch = self
            .sessions
            .query(session, query)
            .map_err(invalid_status)?;
        let input = stream::iter(vec![Ok(batch)]);
        let output = FlightDataEncoderBuilder::new()
            .build(input)
            .map(|result| result.map_err(|error| Status::internal(error.to_string())));
        Ok(Response::new(Box::pin(output)))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoPutStream>, Status> {
        let mut input = request.into_inner();
        let first = input
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("DoPut stream is empty"))??;
        let descriptor = first
            .flight_descriptor
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("first FlightData needs a descriptor"))?;
        let session = descriptor
            .path
            .first()
            .ok_or_else(|| Status::invalid_argument("descriptor path needs a session UUID"))
            .and_then(|value| parse_uuid(value))?;
        let flight_stream =
            stream::once(async { Ok(first) }).chain(input.map_err(FlightError::from));
        let mut batches = FlightRecordBatchStream::new_from_flight_data(flight_stream);
        let mut nodes = Vec::new();
        while let Some(batch) = batches.next().await {
            nodes.extend(batch_to_nodes(&batch.map_err(invalid_status)?)?);
        }
        self.sessions
            .ingest(session, nodes)
            .map_err(invalid_status)?;
        Ok(Response::new(Box::pin(stream::iter(vec![Ok(
            PutResult::default(),
        )]))))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> std::result::Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("DoExchange is not implemented"))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> std::result::Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        let result = match action.r#type.as_str() {
            "MainBranch" => self.sessions.engine().main_branch().id.to_string(),
            "ForkSession" => {
                let parent = parse_uuid_bytes(&action.body)?;
                self.sessions
                    .fork_session(parent)
                    .map_err(invalid_status)?
                    .id
                    .to_string()
            }
            "CommitSession" => {
                let session = parse_uuid_bytes(&action.body)?;
                self.sessions
                    .commit_session(session)
                    .map_err(invalid_status)?
                    .id
                    .to_string()
            }
            "Explain" => {
                let (session, query) = parse_session_payload(&action.body)?;
                self.sessions
                    .explain(session, query)
                    .map_err(invalid_status)?
            }
            other => return Err(Status::unimplemented(format!("unknown action {other}"))),
        };
        Ok(Response::new(Box::pin(stream::iter(vec![Ok(
            ActionResult {
                body: result.into(),
            },
        )]))))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> std::result::Result<Response<Self::ListActionsStream>, Status> {
        Ok(Response::new(Box::pin(stream::iter(
            [
                ("MainBranch", "Return the root branch UUID"),
                ("ForkSession", "Fork a durable branch from a parent UUID"),
                ("CommitSession", "Seal a speculative session"),
                ("Explain", "Return the optimized EnQL physical plan"),
            ]
            .into_iter()
            .map(|(r#type, description)| {
                Ok(ActionType {
                    r#type: r#type.to_owned(),
                    description: description.to_owned(),
                })
            }),
        ))))
    }
}

pub async fn serve_flight(
    address: SocketAddr,
    sessions: Arc<SessionManager>,
) -> std::result::Result<(), tonic::transport::Error> {
    tonic::transport::Server::builder()
        .add_service(FlightServiceServer::new(EngramFlightService::new(sessions)))
        .serve(address)
        .await
}

fn batch_to_nodes(batch: &RecordBatch) -> std::result::Result<Vec<FusedNode>, Status> {
    let keys = column::<StringArray>(batch, "key")?;
    let vectors = column::<FixedSizeListArray>(batch, "vector")?;
    let assertions = column::<UInt64Array>(batch, "assertion_time")?;
    let valid_times = column::<Int64Array>(batch, "valid_time")?;
    if keys.len() != batch.num_rows()
        || vectors.len() != batch.num_rows()
        || assertions.len() != batch.num_rows()
        || valid_times.len() != batch.num_rows()
    {
        return Err(Status::invalid_argument(
            "ingestion columns have inconsistent lengths",
        ));
    }
    (0..batch.num_rows())
        .map(|row| {
            let values = vectors.value(row);
            let values = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Status::invalid_argument("vector items must be float32"))?;
            FusedNode::new(
                keys.value(row),
                vec![TemporalPoint {
                    assertion_time: assertions.value(row),
                    valid_time: valid_times.value(row),
                }],
                values.values().to_vec(),
                Vec::new(),
            )
            .map_err(invalid_status)
        })
        .collect()
}

fn column<'a, T: 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> std::result::Result<&'a T, Status> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| Status::invalid_argument(format!("missing column {name}")))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| Status::invalid_argument(format!("column {name} has the wrong type")))
}

fn parse_session_payload(bytes: &[u8]) -> std::result::Result<(Uuid, &str), Status> {
    let value = std::str::from_utf8(bytes)
        .map_err(|_| Status::invalid_argument("payload must be UTF-8"))?;
    let (session, query) = value
        .split_once('\n')
        .ok_or_else(|| Status::invalid_argument("payload must be SESSION_UUID\\nQUERY"))?;
    Ok((parse_uuid(session)?, query))
}

fn parse_uuid_bytes(bytes: &[u8]) -> std::result::Result<Uuid, Status> {
    let value = std::str::from_utf8(bytes)
        .map_err(|_| Status::invalid_argument("UUID body must be UTF-8"))?;
    parse_uuid(value)
}

fn parse_uuid(value: &str) -> std::result::Result<Uuid, Status> {
    Uuid::parse_str(value).map_err(|_| Status::invalid_argument("invalid UUID"))
}

fn invalid_status(error: impl std::fmt::Display) -> Status {
    Status::invalid_argument(error.to_string())
}
