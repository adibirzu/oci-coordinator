import pytest
from langchain_core.messages import HumanMessage

from src.llm.providers.oca import ChatOCA


def test_chat_oca_instantiation():
    """Verify that ChatOCA can be instantiated."""
    llm = ChatOCA(
        api_url="http://localhost:8080",
        client_id="test_client",
        idcs_url="http://idcs.oracle.com",
    )
    assert llm is not None
    assert llm._llm_type == "oracle_code_assist"


@pytest.mark.asyncio
async def test_chat_oca_generate_failure():
    """Verify generate fails without auth (simulated)."""
    llm = ChatOCA(
        api_url="http://localhost:8080",
        client_id="test_client",
        idcs_url="http://idcs.oracle.com",
    )
    with pytest.raises(RuntimeError, match="OCA authentication required"):
        await llm.ainvoke([HumanMessage(content="Hello")])
