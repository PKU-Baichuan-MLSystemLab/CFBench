import os, random, time
from openai import OpenAI
from httpx import Timeout

class deepseek_v2():
    def __init__(self, model_name="deepseek-chat"):
        self.model_name = model_name
        self.api_key = ''    # api_key
        self.base_url = "https://api.deepseek.com"
        self.client = OpenAI(api_key = self.api_key,base_url = self.base_url)
        print(f"model_name:{self.model_name}")

    def __call__(self, message, maxtry=10):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        i = 0
        response = ""
        while i < maxtry:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": message},
                    ],
                    stream=False
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response

if __name__ == '__main__':
    print(deepseek_v2()("1+1"))   
