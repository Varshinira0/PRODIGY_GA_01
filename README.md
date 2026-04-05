# Generative AI Task 01: Text Generation with GPT-2

This project demonstrates how to fine-tune the GPT-2 transformer model, initially developed by OpenAI, on a custom text dataset to generate contextually relevant text that mimics the style of the training data.

## Description

The objective of this task is to train a model to generate coherent and relevant text based on a given prompt. This is achieved by taking a pre-trained GPT-2 model and fine-tuning it using the Hugging Face `transformers` library with a custom sample dataset.

## Setup Instructions

1.  **Install Constraints & Requirements:** Ensure you have Python installed. The required libraries are listed in `requirements.txt`.
    To install them, run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset Preparation:** A sample `dataset.txt` file is provided in this repository. You can modify this dataset with your custom text if you want the model to mimic a specific style or learn a specific domain.

## Running the Project

To start the fine-tuning process and generate text automatically, run the main script:
```bash
python gpt2_text_generation.py
```

### What the Script Does:

1.  **Fine-tuning (`train_gpt2` function):**
    *   Loads the pre-trained `gpt2` model and tokenizer.
    *   Reads the `dataset.txt`.
    *   Trains the model using the `Trainer` API from the `transformers` library. The script runs for 3 epochs by default but can be customized inside the python script.
    *   Saves the newly fine-tuned custom model and its tokenizer to the `./custom_gpt2_model` directory.

2.  **Text Generation (`generate_text` function):**
    *   Loads the model that was fine-tuned and saved in step 1.
    *   Takes a starting prompt (e.g., "Artificial intelligence will").
    *   Generates text using techniques like Top-K and Top-P (nucleus) sampling, preventing repetitive outputs.
    *   Outputs 3 different variations of the generated sequence based on the prompt.

## Implementation References:
*   [Hugging Face Blog: How to generate text](https://huggingface.co/blog/how-to-generate)
*   [OpenAI GPT-2 Model details](https://openai.com/research/better-language-models/)
