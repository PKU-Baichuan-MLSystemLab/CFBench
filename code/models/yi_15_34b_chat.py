from transformers import AutoModelForCausalLM, AutoTokenizer

class yi_15_34b_chat():
    def __init__(self, model_name="01-ai/Yi-1.5-34B-Chat"):
        self.model_name_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path, use_fast=False)

        # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path,
            device_map="auto",
            torch_dtype='auto'
        ).eval()

    def __call__(self, message):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        # Prompt content: "hi"
        messages = [
            {"role": "user", "content": message}
        ]
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Model response: "Hello! How can I assist you today?"
        return response
    
if __name__ == '__main__':
    print(yi_15_34b_chat()("1+1"))