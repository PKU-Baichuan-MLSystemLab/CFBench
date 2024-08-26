import sys
import json 
import random 
from openai import OpenAI

class gpt4_turbo_0409():
    def __init__(self, model_name="gpt-4-turbo-2024-04-09") -> None:
        self.api_key = ""
        self.base_url = ""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
    print(gpt4_turbo_0409()("1+1"))
    
    
