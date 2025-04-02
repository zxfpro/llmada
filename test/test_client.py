# test_api_clients.py
import pytest
from api_clients import OpenAIClient, AnthropicClient, llamaindex_openaiClient
from unittest.mock import Mock
import requests

@pytest.fixture
def openai_client():
    return OpenAIClient(api_key="test_api_key")

@pytest.fixture
def anthropic_client():
    return AnthropicClient(api_key="test_api_key")

@pytest.fixture
def llamaindex_client():
    return llamaindex_openaiClient(api_key="test_api_key")

def test_openai_client_request(openai_client, monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {"text": "Response from OpenAI"}
    mock_response.raise_for_status.return_value = None

    def mock_post(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(requests, "post", mock_post)

    params = {"model": "gpt-3.5-turbo", "prompt": "Hello, how are you?"}
    response = openai_client.request(params)
    assert response == {"text": "Response from OpenAI"}

def test_anthropic_client_request(anthropic_client, monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {"text": "Response from Anthropic"}
    mock_response.raise_for_status.return_value = None

    def mock_post(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(requests, "post", mock_post)

    params = {"model": "claude-2", "prompt": "Hello, how are you?"}
    response = anthropic_client.request(params)
    assert response == {"text": "Response from Anthropic"}

def test_llamaindex_client_request(llamaindex_client, monkeypatch):
    mock_response = Mock()
    mock_response.json.return_value = {"text": "Response from llamaindex_openai"}
    mock_response.raise_for_status.return_value = None

    def mock_post(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(requests, "post", mock_post)

    params = {"model": "test_model", "prompt": "Hello, how are you?"}
    response = llamaindex_client.request(params)
    assert response == {"text": "Response from llamaindex_openai"}