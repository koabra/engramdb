"""vLLM CacheEngine-compatible storage hooks without importing vLLM."""

from __future__ import annotations

from typing import Any

from ..client import EngramClient


class VllmKvCacheAdapter:
    """Persist and restore canonical vLLM PagedAttention cache bytes.

    A deployment plugin can use ``native_restore_ticket`` with a CUDA extension
    for registered-device GDS reads. CPU-only deployments use ``restore_bytes``.
    """

    def __init__(self, client: EngramClient) -> None:
        self.client = client

    def store(self, session_id: str, spec: dict, cache: Any) -> None:
        self.client.put_kv_cache(_as_bytes(cache), spec, session_id)

    def restore_bytes(self, session_id: str) -> tuple[memoryview, dict]:
        buffer, spec = self.client.get_kv_cache(session_id)
        return memoryview(buffer), spec

    def restore_into(self, session_id: str, destination: Any) -> dict:
        source, spec = self.restore_bytes(session_id)
        target = memoryview(destination).cast("B")
        if len(target) != len(source):
            raise ValueError(
                f"destination has {len(target)} bytes; cache has {len(source)}"
            )
        target[:] = source
        return spec

    def native_restore_ticket(self, session_id: str) -> dict:
        return self.client.get_kv_cache_ticket(session_id)


def _as_bytes(cache: Any) -> bytes:
    if isinstance(cache, bytes):
        return cache
    if isinstance(cache, (bytearray, memoryview)):
        return bytes(cache)
    if hasattr(cache, "detach"):
        cache = cache.detach()
    if hasattr(cache, "contiguous"):
        cache = cache.contiguous()
    if hasattr(cache, "cpu"):
        cache = cache.cpu()
    if hasattr(cache, "numpy"):
        return cache.numpy().tobytes(order="C")
    return memoryview(cache).cast("B").tobytes()
