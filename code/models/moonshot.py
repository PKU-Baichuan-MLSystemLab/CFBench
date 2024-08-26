import requests
from openai import OpenAI
import json

class moonshot():
    def __init__(self, model_name="moonshot-v1-8k"):
        self.model_name = model_name
        self.api_key = ''    # api_key
        self.base_url = "https://api.moonshot.cn/v1"
        self.client = OpenAI(api_key = self.api_key,base_url = self.base_url)
        print(f"model_name:{self.model_name}")

    def __call__(self, message, maxtry=10):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        i = 0
        response = ""
        while i < maxtry:
            try:
                completion = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = [
                        {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                        {"role": "user", "content": message}],
                    temperature = 0.3,
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response


if __name__ == '__main__':
    print(moonshot()("1+1"))