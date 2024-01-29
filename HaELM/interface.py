import os
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

llama_path = "path/to/your/llama/checkpoint"
checkpoint_path = "checkpoint"
data_path = "data/mPLUG_caption.jsonl"

tokenizer = LlamaTokenizer.from_pretrained(llama_path)

model = LlamaForCausalLM.from_pretrained(
    llama_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    checkpoint_path,
    force_download=True,
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)


for d in data:
    image_id = d["image"]
    captions = d["reference caption"]
    prompt = ""
    prompt += "reference captions:\n"
    for caption in captions:
        prompt += caption + ' '
    prompt = prompt[:-1]
    prompt += "\nour caption:\n"
    prompt += d["output"].split('\n')[0].strip()
    prompt += "\nIs our caption accurate?\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1,
        )

    sentence = generation_output.sequences
    sentence = tokenizer.decode(sentence.tolist()[0], skip_special_tokens=True)
    result = sentence.split("\n")[-1]
    print(result)