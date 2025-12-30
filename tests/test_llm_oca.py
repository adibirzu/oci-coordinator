
import pytest
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from src.llm.providers.oca import ChatOCA, OCAAuth
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ChatMessage

def test_chat_oca_instantiation():
    """Verify that ChatOCA can be instantiated."""
    llm = ChatOCA(
        api_url="http://localhost:8080",
        idcs_url="http://idcs.oracle.com",
        client_id="test_client"
    )
    assert llm is not None
    assert llm._llm_type == "oracle_code_assist"

def test_convert_message_to_dict():
    """Verify message conversion logic."""
    llm = ChatOCA(
        api_url="http://localhost:8080",
        idcs_url="http://idcs.oracle.com",
        client_id="test_client"
    )
    
    assert llm._convert_message_to_dict(HumanMessage(content="hi")) == {"role": "user", "content": "hi"}
    assert llm._convert_message_to_dict(AIMessage(content="hello")) == {"role": "assistant", "content": "hello"}
    assert llm._convert_message_to_dict(SystemMessage(content="sys")) == {"role": "system", "content": "sys"}
    assert llm._convert_message_to_dict(ChatMessage(content="custom", role="special")) == {"role": "special", "content": "custom"}
    
    with pytest.raises(ValueError, match="Unknown message type"):
        llm._convert_message_to_dict(MagicMock())

@pytest.mark.asyncio
async def test_chat_oca_generate_failure():
    """Verify generate fails without auth (simulated)."""
    llm = ChatOCA(
        api_url="http://localhost:8080",
        idcs_url="http://idcs.oracle.com",
        client_id="test_client"
    )
    with patch.object(llm._auth, 'get_valid_token', return_value=None):
        with pytest.raises(RuntimeError, match="OCA authentication required"):
            await llm.ainvoke([HumanMessage(content="Hello")])

def test_oca_auth_init():
    """Verify OCAAuth initialization."""
    auth = OCAAuth(idcs_url="http://idcs", client_id="cid", redirect_uri="http://red")
    assert auth.idcs_url == "http://idcs"
    assert auth.client_id == "cid"
    assert auth.redirect_uri == "http://red"

def test_oca_auth_token_expired():
    """Verify token expiration logic."""
    auth = OCAAuth(idcs_url="http://idcs", client_id="cid", redirect_uri="http://red")
    auth._token = {"expires_at": 0}
    assert auth._is_token_expired() is True
    auth._token = {"expires_at": time.time() + 1000}
    assert auth._is_token_expired() is False

def test_oca_auth_load_save_token(tmp_path):
    """Verify token loading and saving."""
    with patch("src.llm.providers.oca.OCAAuth.TOKEN_CACHE_FILE", tmp_path / "token.json"):
        auth = OCAAuth(idcs_url="http://idcs", client_id="cid", redirect_uri="http://red")
        token = {"access_token": "abc", "expires_at": time.time() + 1000}
        auth._save_token(token)
        
        auth2 = OCAAuth(idcs_url="http://idcs", client_id="cid", redirect_uri="http://red")
        assert auth2._token == token

@pytest.mark.asyncio
async def test_oca_auth_get_valid_token_refresh():
    """Verify token refresh logic in get_valid_token."""
    auth = OCAAuth(idcs_url="http://idcs", client_id="cid", redirect_uri="http://red")
    auth._token = {"access_token": "old", "refresh_token": "ref", "expires_at": 0}
    
    async def mock_refresh():
        auth._token = {"access_token": "new", "expires_at": time.time() + 1000}
        return auth._token

    with patch.object(auth, "refresh_token", side_effect=mock_refresh) as mock_refresh_call:
        token = await auth.get_valid_token()
        assert token == "new"
        mock_refresh_call.assert_called_once()

@pytest.mark.asyncio
async def test_chat_oca_agenerate_success():
    """Verify successful generation."""
    llm = ChatOCA(
        api_url="http://localhost:8080",
        idcs_url="http://idcs.oracle.com",
        client_id="test_client"
    )
    
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 10},
        "model": "test-model"
    })
    mock_resp.__aenter__.return_value = mock_resp
    
    with patch.object(llm._auth, "get_valid_token", AsyncMock(return_value="token")):
        with patch("aiohttp.ClientSession.post", return_value=mock_resp):
            res = await llm.ainvoke([HumanMessage(content="hi")])
            assert res.content == "response"

@pytest.mark.asyncio
async def test_oca_auth_refresh_token_api():
    """Verify refresh_token API call."""
    auth = OCAAuth(idcs_url="http://idcs", client_id="cid", redirect_uri="http://red")
    auth._token = {"refresh_token": "ref"}
    
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"access_token": "new", "expires_in": 3600})
    mock_resp.__aenter__.return_value = mock_resp
    
    with patch("aiohttp.ClientSession.post", return_value=mock_resp):
        token = await auth.refresh_token()
        assert token["access_token"] == "new"
        assert auth._token["access_token"] == "new"
