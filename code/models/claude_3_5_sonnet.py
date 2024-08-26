import os, sys, json, time, random
# import anthropic
from httpx import Timeout
from openai import OpenAI

class claude_35_sonnet():
    def __init__(self, model_name="claude-3-5-sonnet-20240620") -> None:
        self.api_key = ""
        self.base_url = ""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        print(f"model_name: {self.model_name}")
    
    def __call__(self, message, temperature=None, maxtry=10):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        messages = [{"role":"user", "content": message}]
        i = 0
        while i < maxtry:
            try:
                if temperature is None:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages,
                        temperature=temperature
                    )
                response = response.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response

if __name__ == "__main__":
    print(claude_35_sonnet()("1+1"))
