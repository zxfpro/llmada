"""
client.py 用于管理client
"""
import requests
import time
import json

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
            time3 = time.time()
            response = requests.post(self.api_base, headers=self.headers, json=params)
            time4 = time.time()
            return response.json()
        except Exception as e:
            raise Exception(f"API request failed: {e}")


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


                print("\n(Streaming finished)") # Add a newline after the stream is complete

            else:
                # Handle non-200 responses
                print(f"Error: Received status code {response.status_code}")
                print("Response body:")
                print(response.text) # Print the full error response if not streaming

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import os

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