# test_api_clients.py
import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
from unittest.mock import Mock
from unittest.mock import patch
from llmada.client import OpenAIClient



# 模拟 OpenAI API 响应
@pytest.fixture
def mock_response():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm fine, thank you!"
                }
            }
        ]
    }

# 模拟流式响应的生成器
def mock_stream_response():
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "I'm"}}]},
        {"choices": [{"delta": {"content": " fine"}}]},
        {"choices": [{"delta": {"content": ", thank you!"}}]},
    ]
    for chunk in chunks:
        yield chunk

# 测试简单对话
@patch("openai.ChatCompletion.create")
def test_request_success(mock_create, mock_response):
    # 模拟成功响应
    mock_create.return_value = mock_response

    client = OpenAIClient(api_key="test_key")
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50
    }

    response = client.request(params)
    assert response["choices"][0]["message"]["content"] == "I'm fine, thank you!"
    mock_create.assert_called_once_with(**params)

# 测试简单对话失败
@patch("openai.ChatCompletion.create")
def test_request_failure(mock_create):
    # 模拟失败响应
    mock_create.side_effect = Exception("API request failed")

    client = OpenAIClient(api_key="test_key")
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50
    }

    with pytest.raises(Exception) as exc_info:
        client.request(params)
    assert str(exc_info.value) == "API request failed"

# 测试流式对话
@patch("openai.ChatCompletion.create")
def test_request_stream_success(mock_create):
    # 模拟流式响应
    mock_create.return_value = mock_stream_response()

    client = OpenAIClient(api_key="test_key")
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Tell me a story about a cat."}],
        "max_tokens": 100
    }

    response = client.request_stream(params)
    content = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        content += delta.get("content", "")
    assert content == "I'm fine, thank you!"

# 测试流式对话失败
@patch("openai.ChatCompletion.create")
def test_request_stream_failure(mock_create):
    # 模拟失败响应
    mock_create.side_effect = Exception("Stream API request failed")

    client = OpenAIClient(api_key="test_key")
    params = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Tell me a story about a cat."}],
        "max_tokens": 100
    }

    with pytest.raises(Exception) as exc_info:
        list(client.request_stream(params))  # 转换生成器为列表以触发异常
    assert str(exc_info.value) == "Stream API request failed"