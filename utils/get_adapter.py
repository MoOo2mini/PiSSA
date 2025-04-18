from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import torch
import os 

parser = argparse.ArgumentParser(description='Merge Adapter to Base Model')
parser.add_argument('--base_model', type=str, default="/home/mhlee/PiSSA/output/metamath-PiSSA-Llama-2-7b-r128_782")
parser.add_argument('--output_path', type=str, default="/home/mhlee/PiSSA/output/metamath-PiSSA-Llama-2-7b-r128_782_j")
args = parser.parse_args()

final_ckpt_dir = os.path.join(args.output_path, "checkpoint-final")
os.makedirs(final_ckpt_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
peft_model = PeftModel.from_pretrained(model, args.output_path)
peft_model.save_pretrained(final_ckpt_dir)
tokenizer.save_pretrained(final_ckpt_dir)