from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm

class deepseek_v2_lite_chat():
    def __init__(self, model_name="deepseek-ai/DeepSeek-V2-Lite-Chat"):
        sefl.max_model_len, self.tp_size = 8192, 8
        self.model_name_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)
        self.llm = LLM(model=self.model_name_path, tensor_parallel_size=tp_size, \
                max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
        self.sampling_params = SamplingParams(temperature=0.3, max_tokens=256, \
                    stop_token_ids=[tokenizer.eos_token_id])

    def __call__(self, message):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        messages_list = [[{"role": "user", "content": message}]]
        prompt_token_ids = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]
        outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)
        outputs = outputs[0].outputs[0].text
        response = outputs
        return response


if __name__ == '__main__':
    print(deepseek_v2_lite_chat()("1+1"))