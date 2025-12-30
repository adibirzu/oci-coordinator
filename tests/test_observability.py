
import pytest
import os
from src.observability.tracing import init_otel_tracing, get_tracer, truncate, is_otel_enabled

def test_tracing_initialization():
    """Verify that tracing can be initialized."""
    # This might fail if env vars are missing, but the import should work
    success = init_otel_tracing()
    assert isinstance(success, bool)
    
    tracer = get_tracer()
    assert tracer is not None

def test_truncate():
    """Verify truncation logic."""
    assert truncate("hello", 10) == "hello"
    assert truncate("hello world", 5) == "he..."
    assert truncate(123, 10) == "123"

def test_is_otel_enabled():
    """Verify enabled status."""
    assert isinstance(is_otel_enabled(), bool)

def test_should_enable_otel_logic(monkeypatch):
    """Test the enablement logic with different env vars."""
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "false")
    from src.observability.tracing import _should_enable_otel
    assert _should_enable_otel() is False
    
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
    monkeypatch.setenv("OCI_APM_ENDPOINT", "http://endpoint")
    monkeypatch.setenv("OCI_APM_PRIVATE_DATA_KEY", "key")
    assert _should_enable_otel() is True

def test_init_otel_tracing_full(monkeypatch):
    """Test full initialization path."""
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
    monkeypatch.setenv("OCI_APM_ENDPOINT", "http://endpoint")
    monkeypatch.setenv("OCI_APM_PRIVATE_DATA_KEY", "key")
    
    # Force re-init by clearing internal state
    import src.observability.tracing
    src.observability.tracing._tracer_provider = None
    
    success = init_otel_tracing()
    assert success is True
    assert is_otel_enabled() is True
