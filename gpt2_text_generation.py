import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

def load_dataset(file_path, tokenizer, block_size=128):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file was not found: {file_path}")
        
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def train_gpt2(train_file_path, model_name="gpt2", output_dir="./models/gpt2-custom"):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load dataset
    train_dataset = load_dataset(train_file_path, tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3, # Increase epochs for better results on a larger dataset
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the trained model and tokenizer
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete and model saved.")

def generate_text(model_path, prompt, max_length=100, num_return_sequences=1):
    # Load fine-tuned model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Ensure PAD token is set
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Encode prompt
    model_inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate text
    print(f"Generating text for prompt: '{prompt}'")
    outputs = model.generate(
        **model_inputs, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2, # Reduce repetition
        do_sample=True, # Sampling
        top_k=50, # Top-k sampling
        top_p=0.95, # Top-p sampling
        temperature=0.7 # Temperature for creativity
    )

    print("\n--- Generated Output ---")
    for i, output in enumerate(outputs):
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Result {i+1}:\n{decoded_text}\n{'.' * 40}")

if __name__ == "__main__":
    dataset_file = "dataset.txt"
    model_output_dir = "./custom_gpt2_model"

    # Step 1: Fine-tune the model
    try:
        train_gpt2(train_file_path=dataset_file, output_dir=model_output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'dataset.txt' exists in the same directory.")
        exit(1)

    # Step 2: Generate text using the fine-tuned model
    prompt = "Artificial intelligence will"
    generate_text(model_path=model_output_dir, prompt=prompt, max_length=50, num_return_sequences=3)
