import pytest
from langchain_anthropic import ChatAnthropic

from src.llm.factory import LLMFactory
from src.llm.oca import ChatOCA


def test_factory_creates_oca():
    """Verify factory creates OCA provider."""
    config = {
        "provider": "oca",
        "model_name": "oca/gpt5",
    }
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatOCA)
    assert llm.model == "oca/gpt5"


def test_factory_creates_anthropic():
    """Verify factory creates Anthropic provider."""
    config = {
        "provider": "anthropic",
        "api_key": "test_key",
        "model_name": "claude-3-sonnet-20240229",
    }
    llm = LLMFactory.create_llm(config)
    assert isinstance(llm, ChatAnthropic)


def test_factory_invalid_provider():
    """Verify factory raises error for invalid provider."""
    config = {"provider": "invalid"}
    with pytest.raises(ValueError, match="Unknown LLM provider: invalid"):
        LLMFactory.create_llm(config)
