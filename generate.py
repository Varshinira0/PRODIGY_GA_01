from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate():
    model_path = "./custom_gpt2_model"
    prompt = "Artificial intelligence will"
    
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    model_inputs = tokenizer(prompt, return_tensors='pt')
    
    print("Generating text...\n")
    outputs = model.generate(
        **model_inputs, 
        max_length=60, 
        num_return_sequences=3,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    with open("output.txt", "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            decoded_text = tokenizer.decode(output, skip_special_tokens=True)
            result_str = f"Result {i+1}:\n{decoded_text}\n{'-' * 40}\n"
            print(result_str)
            f.write(result_str)

if __name__ == "__main__":
    generate()
