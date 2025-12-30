
import pytest
from src.llm.factory import LLMFactory
from src.llm.providers.oca import ChatOCA
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

def test_factory_creates_oca():
    """Verify factory creates OCA provider."""
    config = {
        "provider": "oca",
        "api_url": "http://localhost",
        "idcs_url": "http://idcs",
        "client_id": "test"
    }
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatOCA)

def test_factory_creates_anthropic():
    """Verify factory creates Anthropic provider."""
    config = {
        "provider": "anthropic",
        "api_key": "test_key",
        "model_name": "claude-3-sonnet-20240229"
    }
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatAnthropic)

def test_factory_invalid_provider():
    """Verify factory raises error for invalid provider."""
    config = {"provider": "invalid"}
    with pytest.raises(ValueError, match="Unknown LLM provider: invalid"):
        LLMFactory.create_llm(config)
