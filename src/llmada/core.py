"""
模型适配器
"""
from .client import OpenAIClient
from openai import OpenAI
import os
from volcenginesdkarkruntime import Ark
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai.utils import to_openai_message_dicts
from google.genai import types
from google import genai
from dotenv import load_dotenv
from abc import ABC
load_dotenv()

class ResourceExhaustedError(Exception):
    """Raised when a resource's quota has been exceeded."""
    def __init__(self, message="Token quota has been exhausted."):
        self.message = message
        super().__init__(self.message)

class ModelAdapter(ABC):
    """语言大模型的抽象类

    """
    def __init__(self):
        self.api_key = None
        self.temperature = 0.7
        self.client = None
        self.model_name = None
        self.model_pool = []

    def set_model(self, model_name: str):
        """实例化以后用以修改调用的模型, 要求该模型存在于 model_pool 中

        Args:
            model_name (str): _description_
        """
        assert model_name in self.model_pool
        self.model_name = model_name
    
    def set_temperature(self, temperature:float):
        """用于设置模型的temperature

        Args:
            temperature (float): model temperature
        """
        self.temperature = temperature

    def get_model(self)->list[str]:
        """获得当前的 model_pool

        Returns:
            list[str]: models
        """
        return self.model_pool

    def product(self, prompt: str) -> str:
        """子类必须实现的类, 用于和大模型做非流式交互

        Args:
            prompt (str): 提示词

        Raises:
            NotImplementedError: pass

        Returns:
            str: 大模型返回的结果
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def chat(self, messages: list) -> str:
        """子类必须实现的类, 用于和大模型做聊天交互

        Returns:
            str: 大模型返回的结果
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class ModelModalAdapter(ModelAdapter):

    def get_modal_model(self):
        """获取支持多模态的模型列表

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def product_modal(self, prompt: RichPromptTemplate) -> str:
        """提供多模态的非流式交流

        Args:
            prompt (RichPromptTemplate): llama-index 的富文本格式

        Returns:
            str: _description_
        """
        raise NotImplementedError("This method should be implemented by subclasses")

class BianXieAdapter(ModelModalAdapter):
    """BianXie格式的适配器
    """
    def __init__(self, api_key: str = None, api_base: str = "https://api.bianxie.ai/v1/chat/completions"):
        """初始化
        正常情况下, 这两个参数都不需要传入, 而是会自动寻找环境变量,除非要临时改变api_key.
            api_base 不需要特别指定
        Args:
            api_key (str): API key for authentication.
            api_base (str): Base URL for the API endpoint.
        """
        super().__init__()
        self.client = OpenAIClient(api_key=api_key or os.getenv('BIANXIE_API_KEY') , api_base=api_base)
        self.model_pool = ['gemini-2.5-flash-preview-04-17-nothinking',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-0125',
                'gpt-4.1',
                'gpt-4.1-2025-04-14',
                'gpt-4.1-mini',
                'gpt-4.1-mini-2025-04-14',
                'gpt-4.1-nano',
                'gpt-4.1-nano-2025-04-14',
                'gpt-4o',
                'gpt-4o-2024-11-20',
                'gpt-4o-audio-preview',
                'gpt-4o-audio-preview-2024-10-01',
                'gpt-4o-audio-preview-2024-12-17',
                'gpt-4o-all',
                'gpt-4o-image',
                'gpt-4o-image-vip',
                'gpt-4o-mini',
                'gpt-4o-mini-2024-07-18',
                'gpt-4o-mini-audio-preview',
                'gpt-4o-mini-audio-preview-2024-12-17',
                'gpt-4o-mini-realtime-preview',
                'gpt-4o-mini-realtime-preview-2024-12-17',
                'gpt-4o-mini-search-preview',
                'gpt-4o-mini-search-preview-2025-03-11',
                'gpt-4o-realtime-preview',
                'gpt-4o-realtime-preview-2024-10-01',
                'gpt-4o-realtime-preview-2024-12-17',
                'gpt-4o-search-preview-2025-03-11',
                'gpt-4o-search-preview',
                'claude-3-5-haiku-20241022',
                'claude-3-5-haiku-latest',
                'claude-3-5-sonnet-20240620',
                'claude-3-5-sonnet-20241022',
                'claude-3-5-sonnet-20241022-all',
                'claude-3-5-sonnet-all',
                'claude-3-5-sonnet-latest',
                'claude-3-7-sonnet-20250219',
                'claude-3-7-sonnet-20250219-thinking',
                'claude-3-haiku-20240307',
                'claude-opus-4-20250514',
                'claude-opus-4-20250514-thinking',
                'claude-sonnet-4-20250514',
                'claude-sonnet-4-20250514-thinking',
                'coder-claude3.5-sonnet',
                'coder-claude3.7-sonnet',
                'gemini-2.0-flash',
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash-exp-image-generation',
                'gemini-2.0-flash-thinking-exp',
                'gemini-2.0-flash-thinking-exp-01-21',
                'gemini-2.0-pro-exp-02-05',
                'gemini-2.5-flash-preview-04-17',
                'gemini-2.5-flash-preview-04-17-thinking',
                'gemini-2.5-pro-exp-03-25',
                'gemini-2.5-pro-preview-03-25',
                'gemini-2.5-pro-preview-03-25-thinking',
                'gemini-2.5-pro-preview-05-06',
                'gemini-2.5-pro-preview-05-06-thinking',
                'gemini-2.5-pro-preview-06-05',
                'gemini-2.5-pro-preview-06-05-thinking',
                'deepseek-ai/DeepSeek-R1',
                'deepseek-ai/DeepSeek-V3',
                'deepseek-chat',
                'deepseek-r1',
                'deepseek-r1-250528',
                'deepseek-reasoner',
                'grok-3',
                'grok-3-beta',
                'grok-3-deepsearch',
                'grok-3-mini-beta',
                'grok-3-fast-beta',
                'grok-3-mini-fast-beta',
                'grok-3-reasoner',
                'grok-beta',
                'grok-vision-beta',
                'o1-mini',
                'o1-mini-2024-09-12',
                'o3-mini',
                'o3-mini-2025-01-31',
                'o3-mini-all',
                'o3-mini-high',
                'o3-mini-low',
                'o3-mini-medium',
                'o4-mini',
                'o4-mini-2025-04-16',
                'o4-mini-high',
                'o4-mini-medium',
                'o4-mini-low',
                'text-embedding-ada-002',
                'text-embedding-3-small',
                'text-embedding-3-large']
        self.model_name = self.model_pool[0]
        self.chat_history = []
    
    def get_modal_model(self)->list[str]:
        """返回多模态模型池

        Returns:
            list[str]: 大量模型的字符串名称列表
        """
        return ['gpt-4o',
                'claude-3-5-sonnet-20240620',
                'claude-3-5-sonnet-20241022',
                'claude-3-5-sonnet-20241022-all',
                'claude-3-5-sonnet-all',
                'claude-3-5-sonnet-latest',
                'claude-3-7-sonnet-20250219',
                'claude-3-7-sonnet-20250219-thinking',]
    
    def product_modal(self, prompt: RichPromptTemplate) -> str:
        assert self.model_name in self.get_modal_model()
        messages = to_openai_message_dicts(prompt)
        data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature
        }
        return self.client.request(data).get('choices')[0].get('message').get('content')

    def _deal_response(self,response):
        """
        处理事件的相应
        """

        # {'error': {'message': '[sk-tQ1***5A6] 该令牌额度已用尽 !token.UnlimitedQuota && token.RemainQuota = -17334 (request id: 20250625111613621934640lckfLdQI)', 'type': 'new_api_error'}} self.client.request(data)

        # {'id': 'chatcmpl-20250625111810984333612YPGvZg5m', 
        # 'model': 'gemini-2.5-flash-preview-04-17', 
        # 'object': 'chat.completion', 
        # 'created': 1750821491, 
        # 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Hello there! How can I help you today?'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 4, 'completion_tokens': 10, 'total_tokens': 14, 'prompt_tokens_details': {'cached_tokens': 0, 'text_tokens': 4, 'audio_tokens': 0, 'image_tokens': 0}, 'completion_tokens_details': {'text_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0}, 'input_tokens': 0, 'output_tokens': 0, 'input_tokens_details': None}} self.client.request(data)


        choices = response.get('choices')
        if choices:
            content = choices[0].get('message').get('content')
            return content
        else:
            raise ResourceExhaustedError(f"{response.get('error')}")


    def product(self, prompt: str) -> str:
        """Generate a response from the model based on a single prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The response generated by the model.
        """
        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature
        }
        response = self.client.request(data)
        return self._deal_response(response=response)
    
        
    def _assert_prompt(self,prompt):
        try:
            assert prompt != ""
            return True
        except AssertionError:
            return False
        
    def product_stream(self, prompt: str) -> str:
        """Generate a response from the model based on a single prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The response generated by the model.
        """
        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            "stream": True
        }
        if not self._assert_prompt(prompt):
            yield ""
        else:
            result_stream = self.client.request_stream_http2(data)
            for i in result_stream:
                yield i



    async def aproduct_stream(self, prompt: str) -> str:
        """Generate a response from the model based on a single prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The response generated by the model.
        """
        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            "stream": True
        }
        async for i in self.client.request_stream_http2_async(data):
            yield i

    def chat(self, messages: list) -> str:
        """Engage in a conversation with the model using a list of messages.

        Args:
            messages (list): A list of message dictionaries, each containing a role and content.

        Returns:
            str: The response generated by the model for the conversation.
        """
        data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature
        }
        try:
            return self.client.request(data).get('choices')[0].get('message').get('content')
        except TypeError as e:
            return e

    def chat_stream(self, messages: list) -> str:
        """Engage in a conversation with the model using a list of messages.

        Args:
            messages (list): A list of message dictionaries, each containing a role and content.

        Returns:
            str: The response generated by the model for the conversation.
        """

        data = {
            "model": self.model_name,
            "messages": messages,
            'temperature': self.temperature,
            "stream": True
        }
        result_stream = self.client.request_stream_http2(data)
        
        for i in result_stream:
            yield i

    def chat_stream_history(self, prompt: str, system:str = '') -> str:
        """Engage in a conversation with the model using a list of messages.

        Args:
            messages (list): A list of message dictionaries, each containing a role and content.

        Returns:
            str: The response generated by the model for the conversation.
        """
        if self.chat_history == [] and system != '':
            self.chat_history.append({'role':'system','content':system})
        self.chat_history.append({'role':'user','content':prompt})
        data = {
            "model": self.model_name,
            "messages": self.chat_history,
            'temperature': self.temperature,
            "stream": True
        }
        
        result_stream = self.client.request_stream_http2(data)

        result_str = ''
        for i in result_stream:
            result_str += i
            yield i
        self.chat_history.append({'role':'assistant','content':result_str})




# 

class ArkAdapter(ModelAdapter):
    def __init__(self, api_key: str = None, api_base: str = None,):
        """初始化

        Args:
            api_key (str): API key for authentication.
            api_base (str): Base URL for the API endpoint.
        """
        super().__init__()

        self.client = Ark(api_key=api_key or os.getenv('ARK_API_KEY'))
        self.model_pool = ["doubao-1-5-pro-256k-250115"]
        self.model_name = self.model_pool[0]

    def product(self, prompt: str) -> str:
        """Generate a response from the model based on a single prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The response generated by the model.
        """
        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature
        }
        return self.client.chat.completions.create(**data).choices[0].message.content

    def chat(self, messages: list) -> str:
        """Engage in a conversation with the model using a list of messages.

        Args:
            messages (list): A list of message dictionaries, each containing a role and content.

        Returns:
            str: The response generated by the model for the conversation.
        """
        data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature
        }
        return self.client.chat.completions.create(**data).choices[0].message.content

class GoogleAdapter(ModelAdapter):
    def __init__(self, api_key: str = None):
        """初始化

        Args:
            api_key (str): API key for authentication.
        """
        super().__init__()
        
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=self.api_key)
        self.model_pool = ["gemini-2.5-flash-preview-04-17"]
        self.model_name = self.model_pool[0]
        self.chat_session = None

    def product(self, prompt: str) -> str:
        """Generate a response from the model based on a single prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The response generated by the model.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                # system_instruction="You are a cat. Your name is Neko.",
                temperature=self.temperature),
            contents=[prompt]
        )
        return response.text

    def chat(self, messages: list) -> str:
        """Engage in a conversation with the model using a list of messages.

        Args:
            messages (list): A list of message dictionaries, each containing a role and content.

        Returns:
            str: The response generated by the model for the conversation.
        """
        # Create a new chat session if one doesn't exist
        if self.chat_session is None:
            self.chat_session = self.client.chats.create(model=self.model_name)
            
        # Get the latest user message (usually the last one in the list)
        latest_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                latest_message = msg.get('content', '')
                break
                
        if latest_message is None:
            return "No user message found"
            
        # Send the message to the chat session
        response = self.chat_session.send_message(
            latest_message,
            config=types.GenerateContentConfig(
                # system_instruction="You are a cat. Your name is Neko.",
                temperature=self.temperature),
        )
        
        return response.text

class KimiAdapter(ModelAdapter):
    """Kimi格式的适配器

    """
    def __init__(self, api_key: str = None, api_base: str = "https://api.moonshot.cn/v1",):
        """初始化

        Args:
            api_key (str): API key for authentication.
            api_base (str): Base URL for the API endpoint.
        """
        super().__init__()
        
        self.client = OpenAI(api_key=api_key or os.getenv('MOONSHOT_API_KEY') , base_url=api_base)
        self.model_pool = ["moonshot-v1-128k","moonshot-v1-128k","moonshot-v1-128k"]
        self.model_name = self.model_pool[0]

    def product(self, prompt: str) -> str:
        """Generate a response from the model based on a single prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The response generated by the model.
        """

        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature
        }
        return self.client.chat.completions.create(**data).choices[0].message.content

    def chat(self, messages: list) -> str:
        """Engage in a conversation with the model using a list of messages.

        Args:
            messages (list): A list of message dictionaries, each containing a role and content.

        Returns:
            str: The response generated by the model for the conversation.
        """
        data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
            'stream': True #
        }
        return self.client.chat.completions.create(**data)