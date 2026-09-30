"""Small PyArrow Flight wrapper for EngramDB sessions and EnQL."""

from __future__ import annotations

import json

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

    def hardware_capabilities(self) -> dict:
        return json.loads(self._action("HardwareCapabilities"))

    def put_kv_cache(self, cache: bytes | bytearray | memoryview, spec: dict, session_id: str) -> None:
        table = pa.table(
            {
                "cache": pa.array([bytes(cache)], type=pa.binary()),
                "spec_json": pa.array([json.dumps(spec)], type=pa.string()),
            }
        )
        descriptor = flight.FlightDescriptor.for_path("kv-cache", session_id)
        writer, _ = self._client.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.done_writing()

    def get_kv_cache(self, session_id: str) -> tuple[pa.Buffer, dict]:
        table = self._client.do_get(
            flight.Ticket(f"KV\n{session_id}".encode("utf-8"))
        ).read_all()
        if table.num_rows != 1:
            raise RuntimeError(f"GetKVCache returned {table.num_rows} rows")
        return table["cache"][0].as_buffer(), json.loads(table["spec_json"][0].as_py())

    def get_kv_cache_manifest(self, session_id: str) -> dict:
        return json.loads(self._action("GetKVCacheManifest", session_id))

    def get_kv_cache_ticket(self, session_id: str) -> dict:
        return json.loads(self._action("GetKVCacheTicket", session_id))

    def query_pandas(self, query: str, session_id: str):
        return self.query(query, session_id).to_pandas()

    def query_polars(self, query: str, session_id: str):
        import polars as pl

        return pl.from_arrow(self.query(query, session_id)).lazy()
