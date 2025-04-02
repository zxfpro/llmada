"""
模型适配器
"""

class ModelAdapter:
    def __init__(self):
        self.api_key = None
        self.model_name = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def set_model(self, model_name: str):
        self.model_name = model_name

    def call_api(self, params: dict) -> any:
        raise NotImplementedError("This method should be implemented by subclasses")

    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

    def chat(self, messages: list) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")


class OpenAIAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.client = OpenAIClient()

    def call_api(self, params: dict) -> any:
        return self.client.request(params)

    def generate_text(self, prompt: str) -> str:
        params = {"prompt": prompt, "model": self.model_name}
        return self.call_api(params)

    def chat(self, messages: list) -> str:
        params = {"messages": messages, "model": self.model_name}
        return self.call_api(params)


class AnthropicAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.client = AnthropicClient()

    def call_api(self, params: dict) -> any:
        return self.client.request(params)

    def generate_text(self, prompt: str) -> str:
        params = {"prompt": prompt, "model": self.model_name}
        return self.call_api(params)


class llamaindex_openaiAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.client = llamaindex_openaiClient()

    def call_api(self, params: dict) -> any:
        return self.client.request(params)

    def generate_text(self, prompt: str) -> str:
        params = {"prompt": prompt, "model": self.model_name}
        return self.call_api(params)


