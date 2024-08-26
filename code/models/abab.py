import requests
import json

class abab():
    def __init__(self, model_name="abab6.5-chat"):
        self.model_name = model_name
        self.group_id=""
        self.api_key=""
        self.url = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=" + self.group_id
        self.headers = {"Content-Type":"application/json", "Authorization":"Bearer " + self.api_key}

    def __call__(self, message, maxtry=10):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        payload = {
            "bot_setting":[
                {
                    "bot_name":"MM智能助理",
                    "content":"MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。",
                }
            ],
            "messages":[{"sender_type":"USER", "sender_name":"小明", "text":message}],
            "reply_constraints":{"sender_type":"BOT", "sender_name":"MM智能助理"},
            "model":"abab6.5-chat",
            "tokens_to_generate":2048,
            "temperature":0.01,
            "top_p":0.95,
        }

        response = ""
        i = 0
        while i < maxtry:
            try:
                response = requests.request("POST", self.url, headers=self.headers, json=payload)
                assert response.status_code == 200
                response = json.loads(response.text)
                response = response["reply"]
                return response
            except Exception as e:
                print(f"try:{i}/{maxtry}\t message:{message}\t Error:{e}", flush=True)
                i += 1
                continue
        return response



if __name__ == '__main__':
    print(abab()("1+1"))