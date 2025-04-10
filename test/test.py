from dotenv import load_dotenv



import os

load_dotenv()
print(os.getenv("BIANXIE_BASE2"))
print(os.environ.get('BIANXIE_BASE2'))

