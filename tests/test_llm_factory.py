import pytest
from langchain_anthropic import ChatAnthropic

from src.llm.factory import LLMFactory
from src.llm.oca import ChatOCA
from src.llm.rate_limiter import RateLimitedLLM


def test_factory_creates_oca():
    """Verify factory creates OCA provider (with rate limiting by default)."""
    config = {
        "provider": "oca",
        "model_name": "oca/gpt5",
    }
    llm = LLMFactory.create_llm(config)
    # With rate limiting enabled (default), returns wrapped LLM
    assert isinstance(llm, RateLimitedLLM)
    assert isinstance(llm._llm, ChatOCA)
    assert llm._llm.model == "oca/gpt5"


def test_factory_creates_oca_without_rate_limiting():
    """Verify factory creates OCA provider without rate limiting."""
    config = {
        "provider": "oca",
        "model_name": "oca/gpt5",
    }
    llm = LLMFactory.create_llm(config, enable_rate_limiting=False)
    assert isinstance(llm, ChatOCA)
    assert llm.model == "oca/gpt5"


def test_factory_creates_anthropic():
    """Verify factory creates Anthropic provider (with rate limiting by default)."""
    config = {
        "provider": "anthropic",
        "api_key": "test_key",
        "model_name": "claude-3-sonnet-20240229",
    }
    llm = LLMFactory.create_llm(config)
    # With rate limiting enabled (default), returns wrapped LLM
    assert isinstance(llm, RateLimitedLLM)
    assert isinstance(llm._llm, ChatAnthropic)


def test_factory_creates_anthropic_without_rate_limiting():
    """Verify factory creates Anthropic provider without rate limiting."""
    config = {
        "provider": "anthropic",
        "api_key": "test_key",
        "model_name": "claude-3-sonnet-20240229",
    }
    llm = LLMFactory.create_llm(config, enable_rate_limiting=False)
    assert isinstance(llm, ChatAnthropic)


def test_factory_invalid_provider():
    """Verify factory raises error for invalid provider."""
    config = {"provider": "invalid"}
    with pytest.raises(ValueError, match="Unknown LLM provider: invalid"):
        LLMFactory.create_llm(config)
