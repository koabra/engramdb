"""Small PyArrow Flight wrapper for EngramDB sessions and EnQL."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.flight as flight


class EngramClient:
    def __init__(self, location: str = "grpc://127.0.0.1:50051") -> None:
        self._client = flight.connect(location)

    def _action(self, action_type: str, body: str = "") -> str:
        results = list(
            self._client.do_action(
                flight.Action(action_type, body.encode("utf-8"))
            )
        )
        if len(results) != 1:
            raise RuntimeError(f"{action_type} returned {len(results)} results")
        return results[0].body.to_pybytes().decode("utf-8")

    def main_branch(self) -> str:
        return self._action("MainBranch")

    def fork_session(self, parent_id: str) -> str:
        return self._action("ForkSession", parent_id)

    def commit_session(self, session_id: str) -> str:
        return self._action("CommitSession", session_id)

    def explain(self, query: str, session_id: str) -> str:
        return self._action("Explain", f"{session_id}\n{query}")

    def query(self, query: str, session_id: str) -> pa.Table:
        ticket = flight.Ticket(f"{session_id}\n{query}".encode("utf-8"))
        return self._client.do_get(ticket).read_all()

    def ingest(self, table: pa.Table, session_id: str) -> None:
        descriptor = flight.FlightDescriptor.for_path(session_id)
        writer, _ = self._client.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.done_writing()

    def query_pandas(self, query: str, session_id: str):
        return self.query(query, session_id).to_pandas()

    def query_polars(self, query: str, session_id: str):
        import polars as pl

        return pl.from_arrow(self.query(query, session_id)).lazy()
