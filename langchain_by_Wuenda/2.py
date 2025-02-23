import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从环境变量中获取 API_KEY 和 BASE_URL
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 确认 API_KEY 和 BASE_URL 已正确加载
if not API_KEY or not BASE_URL:
    raise ValueError("API_KEY 或 BASE_URL 未正确加载")

# 创建 ChatOpenAI 实例
chat = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.01,
    max_tokens=512,
)

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Knock knock."),
    AIMessage(content="Who's there?"),
    HumanMessage(content="Orange"),
]

res = chat.invoke(messages)
res