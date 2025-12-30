
def test_imports():
    """Verify critical dependencies are installed."""
    import langchain
    import langgraph
    import oci
    import fastapi
    import redis
    import pydantic
    import opentelemetry
    import tenacity
    import httpx
    
    assert langchain is not None
    assert langgraph is not None
    assert oci is not None
    assert fastapi is not None
    assert redis is not None
    assert pydantic is not None
    assert opentelemetry is not None
    assert tenacity is not None
    assert httpx is not None
