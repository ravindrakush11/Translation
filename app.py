import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set up the Hugging Face token
hf_token = ""  # Alternatively, use os.getenv("HUGGINGFACE_TOKEN")
login(hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to create prompt - Iterates through the messages, concatenates them into a single formatted string with appropriate tokens, and returns the formatted string.
def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text

# Function for inference -
# Formats the input prompts using create_prompt_with_chat_format.
# Encodes the prompts into tensors using the tokenizer.
# Runs the model's generate method to produce outputs.
# Decodes the outputs back into strings.
# Strips the input prompt part from the output to get the final response.
# Returns the list of generated outputs.
def inference(input_prompts, model, tokenizer):
    input_prompts = [
        create_prompt_with_chat_format([{"role": "user", "content": input_prompt}], add_bos=False)
        for input_prompt in input_prompts
    ]

    encodings = tokenizer(input_prompts, padding=True, return_tensors="pt")
    encodings = encodings.to(device)

    with torch.inference_mode():
        outputs = model.generate(encodings.input_ids, do_sample=False, max_new_tokens=250)

    output_texts = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=True)

    input_prompts = [
        tokenizer.decode(tokenizer.encode(input_prompt), skip_special_tokens=True) for input_prompt in input_prompts
    ]
    output_texts = [output_text[len(input_prompt) :] for input_prompt, output_text in zip(input_prompts, output_texts)]
    return output_texts

model_name = "ai4bharat/Airavata"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

@app.route('/inference', methods=['POST'])
def run_inference():
    data = request.json
    input_prompts = data.get('input_prompts', [])
    outputs = inference(input_prompts, model, tokenizer)
    return jsonify(outputs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
