import os
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
import sys
import torch
import argparse
from peft import PeftModel
import shutil

def merge_lora(args):
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_file_path, "../")
    sys.path.append(module_path)

    original_model_id = args.original_model_id
    model_id = args.model_id

    print(f"Loading original model from: {original_model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        original_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights from: {model_id}")
    lora_model = PeftModel.from_pretrained(model, model_id)
    merged_model = lora_model.merge_and_unload()

    print(f"Loading tokenizer from checkpoint: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading processor from original model: {original_model_id}")
    processor = AutoProcessor.from_pretrained(original_model_id, trust_remote_code=True)

    if hasattr(processor, 'tokenizer'):
        added_tokens = tokenizer.get_added_vocab()
        if added_tokens:
            print(f"Found {len(added_tokens)} added tokens in checkpoint tokenizer")
            for token in added_tokens.keys():
                if token not in processor.tokenizer.get_vocab():
                    processor.tokenizer.add_tokens([token])
                    print(f"  Added {token} to processor.tokenizer")

    emb_token = "<emb>"
    emb_token_id = None
    if emb_token in tokenizer.get_vocab():
        emb_token_id = tokenizer.convert_tokens_to_ids(emb_token)
        merged_model.config.emb_token_ids = [emb_token_id]
        print(f"Set emb_token_ids in config: {merged_model.config.emb_token_ids}")
        print(f"  <emb> token ID: {emb_token_id}")
    else:
        print(f"Warning: {emb_token} not found in tokenizer vocab!")

    print(f"Saving merged model to: {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    merged_model.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    files_to_copy = [
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json"
    ]

    print("Copying tokenizer files from checkpoint...")
    for file_name in files_to_copy:
        source_file = os.path.join(model_id, file_name)
        target_file = os.path.join(args.save_path, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
            print(f"  Copied {file_name}")
        else:
            print(f"  Skipped {file_name} (not found in checkpoint)")

    print(f"\nModel merged and saved to {args.save_path}")
    print(f"  - Model type: Qwen3VLForConditionalGeneration")
    if emb_token_id:
        print(f"  - Special token: <emb> (ID: {emb_token_id})")
    print(f"  - Tokenizer vocab size: {len(tokenizer)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge LoRA weights for Qwen3-VL')
    parser.add_argument('--original_model_id', type=str, required=True,
                       help='Path to original Qwen3-VL model')
    parser.add_argument('--model_id', type=str, required=True,
                       help='Path to LoRA checkpoint directory')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save merged model')

    args = parser.parse_args()
    merge_lora(args)