import requests
import json
import time
import random
from tqdm import tqdm

class ernie4():
    def __init__(self, model_name="ernie-4.0-8k-0613"):
        self.client_id = ""
        self.client_secret = ""
        self.model_name = model_name
        print(f"model_name:{self.model_name}")
        

    def get_access_token(self):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """ 
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.client_id}&client_secret={self.client_secret}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def __call__(self, message, maxtry=10):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        messages = [{'role': 'user', 'content': message}]

        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-8k-0613?access_token=" + self.get_access_token()        # 0613

        payload = json.dumps({"messages": messages})
        headers = {'Content-Type': 'application/json'}

        i = 0
        response = ""
        while i < maxtry:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                assert response.status_code == 200
                response = json.loads(response.text)
                finish_reason = response["finish_reason"]
                response = response["result"]
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response


if __name__ == '__main__':
    print(ernie4()("1+1"))
