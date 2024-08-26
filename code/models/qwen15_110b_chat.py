from transformers import AutoModelForCausalLM, AutoTokenizer

class qwen15_110b_chat():
    def __init__(self, model_name="Qwen/Qwen1.5-110B-Chat"):
        self.model_name_path = model_name
        self.device = "cuda"    # the device to load the model onto

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)

    def __call__(self, message):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


if __name__ == '__main__':
    print(qwen15_110b_chat()("1+1"))
