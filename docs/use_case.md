```python
%cd ~/GitHub/llmada/src
```

    /Users/zhaoxuefeng/GitHub/llmada/src


# bianxie


```python
from llmada import BianXieAdapter
```


```python
bx = BianXieAdapter()
```


```python
bx.set_model('gpt-3.5-turbo')
```


```python
bx.product('hello')
```

    2.2560229301452637





    'Hello! How can I assist you today?'




```python
for i in bx.product_stream('hello'):
    print(i, end="")
```

    Received streaming response:
    Hello! How can I assist you today?
    (Streaming finished) 1.8848471641540527



```python
bx.chat([{'role':'user','content':'你好'}])
```

    1.8273937702178955





    '你好！有什么可以帮助你的吗？'




```python
for i in bx.chat_stream_history('hello'):
    print(i, end="")
```

    Received streaming response:
    Hello! How can I assist you today?
    (Streaming finished) 2.101140022277832



```python
for i in bx.chat_stream([{'role':'user','content':'你好'}]):
    print(i, end="")
```

    Received streaming response:
    你好！有什么可以帮助你的吗？
    (Streaming finished) 1.993128776550293



```python
bx.get_model()
```




    ['gemini-2.5-flash-preview-04-17-nothinking',
     'gpt-3.5-turbo',
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




```python
bx.get_modal_model()
```




    ['gpt-4o']




```python
# TODO
product_modal
chat_modal
product_stream_modal
chat_stream_modal
chat_stream_history_modal
```


```python
from llmada import set_llama_index
```


```python
set_llama_index(llm_config = {"api_key":"123423"})
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[17], line 1
    ----> 1 set_llama_index(llm_config = {"api_key":"123423"})


    File ~/GitHub/llmada/src/llmada/core.py:448, in set_llama_index(api_key, api_base, model, temperature, llm_config, embed_config)
        444 from llama_index.embeddings.openai import OpenAIEmbedding
        446 api_key=api_key or os.getenv('BIANXIE_API_KEY')
    --> 448 client = OpenAI(
        449     model=model,
        450     api_base=api_base,
        451     api_key=api_key,
        452     temperature=temperature,
        453     **llm_config
        454 )
        455 embedding = OpenAIEmbedding(api_base=api_base,api_key=api_key,**embed_config)
        456 Settings.embed_model = embedding


    TypeError: llama_index.llms.openai.base.OpenAI() got multiple values for keyword argument 'api_key'



```python

```

## bianxie


```python
from llmada.core import GoogleAdapter
```


```python
model = GoogleAdapter()
```


```python
model.model_name
```




    'gemini-2.5-flash-preview-04-17'




```python
model.product('你好')
```




    '你好！有什么可以帮你的吗？'




```python
model.chat([{'role':'user','content':'你好'}])
```




    '你好！有什么我可以帮您的吗？'



