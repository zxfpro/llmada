# test_model_adapter.py
import pytest
from model_adapter import OpenAIAdapter, AnthropicAdapter, llamaindex_openaiAdapter
from unittest.mock import Mock

@pytest.fixture
def openai_adapter():
    adapter = OpenAIAdapter()
    adapter.set_api_key("test_api_key")
    adapter.set_model("gpt-3.5-turbo")
    return adapter

@pytest.fixture
def anthropic_adapter():
    adapter = AnthropicAdapter()
    adapter.set_api_key("test_api_key")
    adapter.set_model("claude-2")
    return adapter

@pytest.fixture
def llamaindex_adapter():
    adapter = llamaindex_openaiAdapter()
    adapter.set_api_key("test_api_key")
    adapter.set_model("test_model")
    return adapter

def test_openai_adapter_generate_text(openai_adapter):
    openai_adapter.client = Mock()
    openai_adapter.client.request.return_value = {"text": "Test response"}
    response = openai_adapter.generate_text("Hello, how are you?")
    assert response == "Test response"

def test_openai_adapter_chat(openai_adapter):
    openai_adapter.client = Mock()
    openai_adapter.client.request.return_value = {"text": "Test response"}
    response = openai_adapter.chat([{"role": "user", "content": "Hello"}])
    assert response == "Test response"

def test_anthropic_adapter_generate_text(anthropic_adapter):
    anthropic_adapter.client = Mock()
    anthropic_adapter.client.request.return_value = {"text": "Test response"}
    response = anthropic_adapter.generate_text("Hello, how are you?")
    assert response == "Test response"

def test_llamaindex_adapter_generate_text(llamaindex_adapter):
    llamaindex_adapter.client = Mock()
    llamaindex_adapter.client.request.return_value = {"text": "Test response"}
    response = llamaindex_adapter.generate_text("Hello, how are you?")
    assert response == "Test response"