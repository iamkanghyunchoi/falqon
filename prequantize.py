import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from torchao.float8 import convert_to_float8_training, convert_to_float8_training_falqon
from transformers import LlamaTokenizer

# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the last module
    if fqn == "1":
        return False
    # if "lm_head" in fqn:
    #     return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def convert_and_save_model(model_name, output_dir, quant_type="", lora_r=32, lora_alpha=16, num_topk=10):
    """
    Load model in CPU, convert to specified quantization type, and save
    Args:
        model_name: Original HF model name/path
        output_dir: Where to save the converted model
        quant_type: Quantization type (default "fp4")
        lora_r: LoRA rank (default 8)
        lora_alpha: LoRA alpha (default 16)
        num_topk: Number of top-k values for partial reset (default 1)
    """
    print(f"Loading model {model_name} in CPU...")
    
    # Load in CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu", 
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    DEFAULT_PAD_TOKEN = "[PAD]"
    smart_tokenizer_and_embedding_resize(dict(pad_token=DEFAULT_PAD_TOKEN), tokenizer, model)


    if 'llama' in model_name.lower() or isinstance(tokenizer, LlamaTokenizer):
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    print(f"Converting model with {quant_type} quantization...")
    

    if not quant_type in ["fp4", "nf4"]:
        if 'falqon' in quant_type:
            print("SVD start")
            import time
            start_time = time.time()
            convert_to_float8_training_falqon(model, module_filter_fn=module_filter_fn, rank=args.lora_r, lora_alpha=args.lora_alpha, num_topk=args.num_topk)
            end_time = time.time()
            print(f"SVD time: {end_time - start_time} seconds")
    
    print(f"Saving converted model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    
    tokenizer.save_pretrained(output_dir, safe_serialization=False)
    
    print("Done!")
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert and save model with quantization')
    parser.add_argument('--model_name', type=str, required=True, help='Original HF model name/path')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save the converted model')
    parser.add_argument('--quant_type', type=str, default="", help='Quantization type (default: "")')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank (default: 32)')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha (default: 16)')
    parser.add_argument('--num_topk', type=int, default=10, help='Number of top-k values for partial reset (default: 10)')
    
    args = parser.parse_args()
    
    convert_and_save_model(
        args.model_name,
        args.output_dir,
        args.quant_type,
        args.lora_r,
        args.lora_alpha,
        args.num_topk
    )
