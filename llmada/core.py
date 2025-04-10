"""
模型适配器
"""
from .client import OpenAIClient

class ModelAdapter:
    def __init__(self):
        self.api_key = None
        self.model_name = None
        self.temperature = 0.7

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def set_model(self, model_name: str):
        self.model_name = model_name

    def product(self, prompt: str) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

    def chat(self, messages: list) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")


class OpenAIAdapter(ModelAdapter):
    def __init__(self,api_key: str, api_base : str):
        super().__init__()
        self.client = OpenAIClient(api_key = api_key,
                                   api_base = "https://api.bianxie.ai/v1/chat/completions" )


    def product(self, prompt: str) -> str:
        data = {
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': self.temperature
            }
        return self.client.request(data)

    def chat(self, messages: list) -> str:
        data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature
            }
        return self.client.request(data)
