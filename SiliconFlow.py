from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens
import requests
import os

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
# print(API_KEY, BASE_URL)

class SiliconFlow(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def siliconflow_completions_prompt(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }

        response = requests.post(BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]
    
    def siliconflow_completions(self, model: str, messages: str, temperature: float = 0.7) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }

        response = requests.post(BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        # print(response.json()["choices"][0]["message"]["content"])
        # print(response.json()["choices"][0]["message"])
        return response.json()["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: list = None, model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", temperature: float = 0.7) -> str:
        response = self.siliconflow_completions_prompt(model=model, prompt=prompt, temperature=temperature)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response
    
    def call(self, messages: list, stop: list = None, model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", temperature: float = 0.7) -> str:
        response = self.siliconflow_completions(model=model, messages=messages, temperature=temperature)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response
    
    @staticmethod
    def HumanMessage(content: str) -> dict:
        return {"role": "user", "content": content}
    
    @staticmethod
    def AIMessage(content: str) -> dict:
        return {"role": "assistant", "content": content}
    
    @staticmethod
    def SystemMessage(content: str) -> dict:
        return {"role": "system", "content": content}

if __name__ == "__main__":
    llm = SiliconFlow()

    messages = [SiliconFlow.SystemMessage(content="You are a helpful assistant."),
                SiliconFlow.HumanMessage(content="Knock knock."),
                SiliconFlow.AIMessage(content="Who's there?"),
                SiliconFlow.HumanMessage(content="Orange"),]

    print(messages)
    response = llm._call(prompt="你是谁？", model="deepseek-ai/DeepSeek-V2.5", temperature=0.9)
    print(response["content"])
    response = llm.call(messages=messages, model="deepseek-ai/DeepSeek-V2.5", temperature=0.9)
    print(response["content"])