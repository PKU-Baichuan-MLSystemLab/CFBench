
from zhipuai import ZhipuAI

class glm4():
    def __init__(self, model_name="glm-4-0520"):
        self.model_name = model_name
        self.api_key = ""       # api_key
        self.client = ZhipuAI(api_key=self.api_key) # 填写您自己的APIKey


    def __call__(self, message, maxtry=10):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        i = 0
        response = ""
        while i < maxtry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  # 填写需要调用的模型编码
                    messages=[
                        {"role": "user", "content": message}
                    ],
                )
                response = response.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response

if __name__ == '__main__':
    print(glm4()("1+1"))