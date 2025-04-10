"""
client.py 用于管理client
"""
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from typing import Any, Dict, Optional
from typing import Dict, Any, Iterator
import openai

import requests

class OpenAIClient:
    """
    使用 openai 官方包对接 OpenAI API，支持简单对话和流式对话
    """
    def __init__(self, api_key: str, api_base: str = "https://api.bianxie.ai/v1/chat/completions"):
        """
        初始化 OpenAI 客户端
        """
        self.api_key = api_key
        self.api_base = api_base
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    def request(self, params: dict) -> dict:
        """
        简单对话：直接调用 OpenAI API 并返回完整响应
        """

        try:
            response = requests.post(self.api_base, headers=self.headers, json=params)
            return response.json()
        except Exception as e:
            raise Exception(f"API request failed: {e}")


class LlamaIndexOpenAIClient:
    """
    使用 llama-index 包对接 OpenAI API，支持简单对话和流式对话
    """
    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1",model="gpt-3.5-turbo"):
        """
        初始化 OpenAI 客户端
        """
        self.api_key = api_key
        self.api_base = api_base

        # 初始化 LLM 和 Embedding 模型
        self.llm = OpenAI(
            model=model,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0.1
        )
        self.embed_model = OpenAIEmbedding(
            api_base=self.api_base,
            api_key=self.api_key
        )
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

    def predict(self, messages: Iterator[ChatMessage], **kwargs: Any) -> ChatMessage:
        """
        简单对话：直接调用 OpenAI API 并返回完整响应
        """
        try:
            response = self.llm.predict(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"API request failed: {e}")