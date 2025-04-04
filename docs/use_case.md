# 使用案例


## 便携模型使用product

```python
from llmada import BianXieAdapter

bianxie = BianXieAdapter(api_key=os.getenv('BIANXIE'))

bianxie.set_model('gpt-4o')

bianxie.product(prompt='你好')
```

## 便携模型使用chat
```python

from llmada import BianXieAdapter

bianxie = BianXieAdapter(api_key=os.getenv('BIANXIE'))

bianxie.set_model('gpt-4o')

bianxie.chat(messages=[{'role': 'user', 'content': 'hello'}],)
```

## kimi模型使用chat
```python
from llmada import KimiAdapter
kimi = KimiAdapter(api_key=os.getenv('MOONSHOT_API_KEY'))
kimi.get_model()
kimi.set_model('moonshot-v1-128k')
kimi.chat(messages=[{'role': 'user', 'content': 'hello'}],)
```

## kimi 模型使用product
```python
from llmada import KimiAdapter
kimi = KimiAdapter(api_key=os.getenv('MOONSHOT_API_KEY'))
kimi.get_model()
kimi.set_model('moonshot-v1-128k')
x = kimi.product(prompt='hello')
```