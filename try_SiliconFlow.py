from SiliconFlow import SiliconFlow

llm = SiliconFlow()

messages = [SiliconFlow.HumanMessage(content="RAG是什么?")]

response = llm.call(messages=messages, model="deepseek-ai/DeepSeek-V2.5", temperature=0.9)
print(response)