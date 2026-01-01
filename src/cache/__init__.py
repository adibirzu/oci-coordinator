"""
OCI Resource Cache Module.

Provides caching layer for OCI resources with Redis backend.

Features:
- Tag-based invalidation for group operations
- Stale-while-revalidate pattern for improved latency
- Event-driven cache updates via pub/sub
- Comprehensive statistics and health checks
"""

from src.cache.oci_resource_cache import (
    CACHE_EVENT_DELETE,
    CACHE_EVENT_EXPIRE,
    CACHE_EVENT_INVALIDATE,
    CACHE_EVENT_SET,
    OCIResourceCache,
)

__all__ = [
    "CACHE_EVENT_DELETE",
    "CACHE_EVENT_EXPIRE",
    "CACHE_EVENT_INVALIDATE",
    "CACHE_EVENT_SET",
    "OCIResourceCache",
]
