"""SGLang RadixAttention KV-cache persistence hooks."""

from __future__ import annotations

from typing import Any

from ..client import EngramClient
from .vllm import _as_bytes


class SglangKvCacheAdapter:
    """Store layer-major SGLang pools or a canonical converted cache."""

    def __init__(self, client: EngramClient) -> None:
        self.client = client

    def export_committed_kv(self, session_id: str, spec: dict, cache: Any) -> None:
        self.client.put_kv_cache(_as_bytes(cache), spec, session_id)

    def restore_radix_prefix(
        self, session_id: str, destination: Any | None = None
    ) -> tuple[memoryview, dict] | dict:
        buffer, spec = self.client.get_kv_cache(session_id)
        source = memoryview(buffer).cast("B")
        if destination is None:
            return source, spec
        target = memoryview(destination).cast("B")
        if len(target) != len(source):
            raise ValueError(
                f"destination has {len(target)} bytes; cache has {len(source)}"
            )
        target[:] = source
        return spec

    def native_restore_ticket(self, session_id: str) -> dict:
        return self.client.get_kv_cache_ticket(session_id)
