"""
client.py 用于管理client
"""
import requests
import time
import json
import base64
import httpx
from .log import Log
logger = Log.logger

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
            time1 = time.time()
            response = requests.post(self.api_base, headers=self.headers, json=params)
            time2 = time.time()
            print(time2-time1)
            return response.json()
        except Exception as e:
            raise Exception(f"API request failed: {e}")



    def request_stream_http2(self, params: dict) -> dict:
        """
        简单对话：直接调用 OpenAI API 并返回流式响应
        """
        try:
            self.headers["Accept"] = "text/event-stream"

            payload = params
            with httpx.Client(http2=True, timeout=None) as client:
                with client.stream("POST", self.api_base, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    buffer = b''
                    for chunk in response.iter_bytes():
                        buffer += chunk
                        while b'\n' in buffer:
                            line_bytes, _, buffer = buffer.partition(b'\n')
                            decoded_line = line_bytes.decode('utf-8')

                            if decoded_line.startswith('data:'):
                                json_str = decoded_line[len('data:'):].strip()

                                if json_str == '[DONE]':
                                    return # End the generator

                                try:
                                    data = json.loads(json_str)
                                    # Extract content based on common LLM API response structure
                                    if "choices" in data and data["choices"]:
                                        chunk_content = data["choices"][0]["delta"].get("content", "")
                                        if chunk_content:
                                            yield chunk_content # Yield the generated content chunk
                                except json.JSONDecodeError:
                                    yield f"[ERROR: Malformed data received: {json_str}]"

        except httpx.RequestError as e:
            error_msg = f"Request failed: {e}"
            raise ConnectionError(error_msg) from e # Re-raise as a custom error or simpler Exception
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            raise RuntimeError(error_msg) from e



    # def request_stream_http2(self, params: dict): # 移除 -> dict 类型提示，因为是生成器
    #     """
    #     简单对话：直接调用 OpenAI API 并返回流式响应
    #     """
        
    #     request_start_time = time.time()
    #     logger.debug(f"[{request_start_time:.3f}] [DEBUG] Request initiation started.")

    #     self.headers["Accept"] = "text/event-stream"
    #     # 确保 User-Agent 设置正确，即使之前测试无效，也让日志记录下来
    #     self.headers["User-Agent"] = "curl/8.6.0" 
        
    #     payload = params

    #     try:
    #         # 客户端初始化时间
    #         client_init_time = time.time()
    #         logger.debug(f"[{client_init_time:.3f}] [DEBUG] Initializing httpx.Client(http2=True).")
            
    #         with httpx.Client(http2=True, timeout=None,verify=False) as client:
    #             client_initialized_time = time.time()
    #             logger.debug(f"[{client_initialized_time:.3f}] [DEBUG] httpx.Client initialized. Time taken: {client_initialized_time - client_init_time:.3f}s.")
                
    #             # 发送请求时间
    #             send_request_time = time.time()
    #             logger.debug(f"[{send_request_time:.3f}] [DEBUG] Sending POST request to {self.api_base}.")
                
    #             with client.stream("POST", self.api_base, headers=self.headers, json=payload) as response:
    #                 first_byte_received_time = time.time()
    #                 logger.debug(f"[{first_byte_received_time:.3f}] [DEBUG] Received first response byte (or headers). Time to first byte from request start: {first_byte_received_time - request_start_time:.3f}s.")
                    
    #                 response.raise_for_status() # 检查HTTP状态码
                    
    #                 # 打印响应头，确认 content-type 和 Server 字段
    #                 # logger.debug(f"[{time.time():.3f}] [DEBUG] Response Status: {response.status_code}, HTTP Version: {response.http_version.value}.")
    #                 logger.debug(f"[{time.time():.3f}] [DEBUG] Response Status: {response.status_code}, HTTP Version: {response.http_version}.")
    #                 logger.debug(f"[{time.time():.3f}] [DEBUG] Response Headers: {response.headers}.")


    #                 buffer = b''
    #                 chunk_count = 0
                    
    #                 for chunk in response.iter_bytes():
    #                     chunk_receive_time = time.time()
    #                     chunk_count += 1
    #                     # 打印每次收到 chunk 的时间
    #                     logger.debug(f"[{chunk_receive_time:.3f}] [DEBUG] Received chunk {chunk_count}. Size: {len(chunk)} bytes.")

    #                     buffer += chunk
    #                     while b'\n' in buffer:
    #                         line_bytes, _, buffer = buffer.partition(b'\n')
    #                         decoded_line = line_bytes.decode('utf-8')
                            
    #                         # 打印每行解析出来的时间
    #                         line_parsed_time = time.time()
    #                         logger.debug(f"[{line_parsed_time:.3f}] [DEBUG] Parsed line: '{decoded_line[:50]}...'") # 打印前50字符避免过长

    #                         if decoded_line.startswith('data:'):
    #                             json_str = decoded_line[len('data:'):].strip()

    #                             if json_str == '[DONE]':
    #                                 logger.debug(f"[{time.time():.3f}] [DEBUG] [DONE] signal received. Stream ending.")
    #                                 return # End the generator

    #                             try:
    #                                 data = json.loads(json_str)
    #                                 # Extract content based on common LLM API response structure
    #                                 if "choices" in data and data["choices"]:
    #                                     chunk_content = data["choices"][0]["delta"].get("content", "")
    #                                     if chunk_content:
    #                                         # Yielding the chunk_content
    #                                         yield_time = time.time()
    #                                         logger.debug(f"[{yield_time:.3f}] [DEBUG] Yielding content: '{chunk_content[:50]}...'")
    #                                         yield chunk_content # Yield the generated content chunk
    #                             except json.JSONDecodeError:
    #                                 logger.error(f"[{time.time():.3f}] [ERROR] Malformed data received: {json_str}", exc_info=True)
    #                                 yield f"[ERROR: Malformed data received: {json_str}]"
                                    
        # except httpx.RequestError as e:
        #     error_msg = f"Request failed: {e}"
        #     logger.error(f"[{time.time():.3f}] [ERROR] httpx.RequestError: {error_msg}", exc_info=True)
        #     raise ConnectionError(error_msg) from e
        # except Exception as e:
        #     error_msg = f"An unexpected error occurred: {e}"
        #     logger.critical(f"[{time.time():.3f}] [CRITICAL] Unexpected error: {error_msg}", exc_info=True)
        #     raise RuntimeError(error_msg) from e


    async def request_stream_http2_async(self, params: dict):
        """
        简单对话：直接调用 OpenAI API 并返回流式响应 (异步版本)
        """
        try:
            # 设置 Accept 头为 text/event-stream，表示期望流式事件
            self.headers["Accept"] = "text/event-stream"
            payload = params

            # 使用 httpx.AsyncClient 进行异步 HTTP 请求
            # timeout=None 表示不设置请求超时时间，由底层 TCP/TLS 决定
            async with httpx.AsyncClient(http2=True, timeout=None) as client:
                # 发送 POST 请求并获取异步响应流
                async with client.stream("POST", self.api_base, headers=self.headers, json=payload) as response:
                    # 检查 HTTP 响应状态码，如果不是 2xx，则抛出异常
                    response.raise_for_status()

                    buffer = b''
                    # 异步迭代响应的字节块
                    async for chunk in response.aiter_bytes():
                        buffer += chunk
                        # 处理可能包含多行或不完整行的缓冲区
                        while b'\n' in buffer:
                            line_bytes, _, buffer = buffer.partition(b'\n')
                            decoded_line = line_bytes.decode('utf-8')

                            if decoded_line.startswith('data:'):
                                json_str = decoded_line[len('data:'):].strip()

                                if json_str == '[DONE]':
                                    return # 结束生成器

                                try:
                                    data = json.loads(json_str)
                                    # 根据常见的 LLM API 响应结构提取内容
                                    if "choices" in data and data["choices"]:
                                        chunk_content = data["choices"][0]["delta"].get("content", "")
                                        if chunk_content:
                                            # 异步 yield 生成的内容块
                                            yield chunk_content
                                except json.JSONDecodeError:
                                    yield f"[ERROR: Malformed data received: {json_str}]"

        except httpx.RequestError as e:
            error_msg = f"Request failed: {e}"
            # 异步函数中也抛出异常
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            # 异步函数中也抛出异常
            raise RuntimeError(error_msg) from e

    def request_stream(self, params: dict) -> dict:
        """
        简单对话：直接调用 OpenAI API 并返回流式响应
        """
        try:
            time1 = time.time()
            response = requests.post(self.api_base, headers=self.headers, json=params, stream=True)

            # 检查HTTP状态码
            if response.status_code == 200:
                print("Received streaming response:")
                # 逐行处理响应内容
                # iter_lines() 迭代响应内容，按行分割，并解码
                time2 = time.time()
                for line in response.iter_lines():
                    # lines are bytes, convert to string
                    line = line.decode('utf-8')

                    # Server-Sent Events (SSE) messages start with 'data: '
                    # and the stream ends with 'data: [DONE]'
                    if line.startswith('data:'):
                        # Extract the JSON string after 'data: '
                        json_str = line[len('data:'):].strip()

                        if json_str == '[DONE]':
                            break # End of stream
                        
                        if json_str: # Ensure it's not an empty data line
                            try:
                                # Parse the JSON string into a dictionary
                                chunk = json.loads(json_str)

                                # Extract the content chunk (similar structure to OpenAI API)
                                # Check if choices and delta exist before accessing content
                                if chunk and 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta')
                                    if delta and 'content' in delta:
                                        content_chunk = delta['content']
                                        # Print the chunk without a newline, immediately flushing the output
                                        # print(content_chunk, end='', flush=True)
                                        yield content_chunk

                            except json.JSONDecodeError as e:
                                print(f"\nError decoding JSON chunk: {e}\nChunk: {json_str}")
                            except Exception as e:
                                print(f"\nError processing chunk: {e}\nChunk data: {chunk}")


                print(f"\n(Streaming finished) {time2-time1}") # Add a newline after the stream is complete

            else:
                # Handle non-200 responses
                print(f"Error: Received status code {response.status_code}")
                print("Response body:")
                print(response.text) # Print the full error response if not streaming

        except requests.exceptions.RequestException as e:
            # TODO if "439" in ccc
            print(f"Request Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")




    def request_modal(self):

        def get_img_content(inputs = str):
            if is_url:
                return "https://github.com/dianping/cat/raw/master/cat-home/src/main/webapp/images/logo/cat_logo03.png"
            else:
                def image_to_base64(image_path):
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                        base64_encoded_data = base64.b64encode(image_data)
                        base64_encoded_str = base64_encoded_data.decode('utf-8')
                        return base64_encoded_str    
                image_base64 = image_to_base64(inputs)
                return f"data:image/jpeg;base64,{image_base64}"


        data = {
            'model': 'gpt-4-vision-preview',
            'messages': [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "这张图片的图标是个什么动物？"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_img_content('111.png')
                            }
                        }
                    ]
                }
            ],
        }

        response = requests.post(self.api_base, headers=self.headers, json=data)
        return response.json()



import os

def test_request():
    opc = OpenAIClient(os.getenv('BIANXIE_API_KEY'))
    data = {
    "model": "gpt-3.5-turbo",
    "messages": [{
        "role": "user",
        "content": "how to deal with world war 3"
    }]
    }
    pp = opc.request(data)
    print(pp)

def test_request_stream():
    opc = OpenAIClient(os.getenv('BIANXIE_API_KEY'))
    data = {
    "model": "gpt-3.5-turbo",
    "messages": [{
        "role": "user",
        "content": "how to deal with world war 3"
    }],
    "stream": True
}
    pp = opc.request_stream(data)
    print(pp)
    for i in pp:
        print(i)

def test_request_modal():
    opc = OpenAIClient(os.getenv('BIANXIE_API_KEY'))
    data = {
    "model": "gpt-3.5-turbo",
    "messages": [{
        "role": "user",
        "content": "how to deal with world war 3"
    }],
    "stream": True
}
    pp = opc.request_stream(data)
    print(pp)
    for i in pp:
        print(i)