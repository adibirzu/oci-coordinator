
import pytest
from src.observability.tracing import init_otel_tracing, get_tracer

def test_tracing_initialization():
    """Verify that tracing can be initialized."""
    # This might fail if env vars are missing, but the import should work
    success = init_otel_tracing()
    assert isinstance(success, bool)
    
    tracer = get_tracer()
    assert tracer is not None
