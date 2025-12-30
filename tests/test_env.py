import fastapi
import httpx
import langchain
import langgraph
import oci
import opentelemetry
import pydantic
import redis
import tenacity


def test_imports():
    """Verify critical dependencies are installed."""
    assert langchain is not None
    assert langgraph is not None
    assert oci is not None
    assert fastapi is not None
    assert redis is not None
    assert pydantic is not None
    assert opentelemetry is not None
    assert tenacity is not None
    assert httpx is not None
