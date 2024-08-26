import requests
from openai import OpenAI
import json

class yi_large():
    def __init__(self, model_name="yi-large"):
        self.model_name = model_name
        self.api_key = ''    # api_key
        self.base_url = "https://api.lingyiwanwu.com/v1"
        self.client = OpenAI(api_key = self.api_key,base_url = self.base_url)
        print(f"model_name:{self.model_name}")

    def __call__(self, message, maxtry=10):
        i = 0
        response = ""
        while i < maxtry:
            try:
                completion = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = [
                        {"role": "user", "content": message}],
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response


if __name__ == '__main__':
    print(yi_large()("1+1"))