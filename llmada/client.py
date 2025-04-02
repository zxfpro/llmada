"""
client.py 用于管理client
"""
import requests
import json
from typing import Any, Dict, Optional

class OpenAIClient:
    def __init__(self, api_key: str, api_base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.api_base_url = api_base_url

    def request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            endpoint = params.get("endpoint", "completions")
            url = f"{self.api_base_url}/{endpoint}"

            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(params)
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")


class AnthropicClient:
    def __init__(self, api_key: str, api_base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self.api_base_url = api_base_url

    def request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            endpoint = params.get("endpoint", "complete")
            url = f"{self.api_base_url}/{endpoint}"

            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(params)
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")


class llamaindex_openaiClient:
    def __init__(self, api_key: str, api_base_url: str = "https://api.llamaindex-openai.com/v1"):
        self.api_key = api_key
        self.api_base_url = api_base_url

    def request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            endpoint = params.get("endpoint", "completions")
            url = f"{self.api_base_url}/{endpoint}"

            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(params)
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")


if __name__ == "__main__":
    # 使用OpenAIClient
    openai_client = OpenAIClient(api_key="your_openai_api_key")
    params = {
        "model": "gpt-3.5-turbo",
        "prompt": "Hello, how are you?",
        "max_tokens": 100
    }
    response = openai_client.request(params)
    print(response)

    # 使用AnthropicClient
    anthropic_client = AnthropicClient(api_key="your_anthropic_api_key")
    params = {
        "model": "claude-2",
        "prompt": "Hello, how are you?",
        "max_tokens": 100
    }
    response = anthropic_client.request(params)
    print(response)

    # 使用llamaindex_openaiClient
    llamaindex_client = llamaindex_openaiClient(api_key="your_llamaindex_api_key")
    params = {
        "model": "test_model",
        "prompt": "Hello, how are you?",
        "max_tokens": 100
    }
    response = llamaindex_client.request(params)
    print(response)