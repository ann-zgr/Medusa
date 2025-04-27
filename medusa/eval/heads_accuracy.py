import os
import torch
import json
from contextlib import contextmanager
import numpy as np
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse
from datasets import load_dataset

def get_accuracies(medusa, logit):
    # get the correct counts of each head
    seq_len, choices, topk = medusa.shape
    results = []
    
    # We only evaluate up to the specified number of medusa heads
    actual_choices = min(choices, args.medusa_num_heads)
    
    for choice in range(actual_choices):
        results.append(medusa[:-choice - 1, choice].eq(logit[choice + 1:, 0]))
    return results

def main(args):
    print(f"Evaluating with {args.medusa_num_heads} Medusa heads")
    
    model = MedusaModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()

    # Load the dataset from HuggingFace
    if args.use_huggingface:
        print(f"Loading dataset from HuggingFace: {args.dataset_name}")
        try:
            # Load the dataset using HuggingFace datasets
            dataset = load_dataset(args.dataset_name)
            
            # Print dataset info
            print(f"Dataset structure: {dataset}")
            
            # Access the main data
            if "train" in dataset:
                data_split = dataset["train"]
                print(f"Using 'train' split with {len(data_split)} samples")
            else:
                # Find the first available split
                split_name = list(dataset.keys())[0]
                data_split = dataset[split_name]
                print(f"Using '{split_name}' split with {len(data_split)} samples")
            
            # Convert to the expected format
            data = []
            for item in data_split:
                if "instruction" in item:
                    data.append({
                        "instruction": item["instruction"]
                    })
            
            # Limit the number of samples if specified
            if args.max_samples and args.max_samples < len(data):
                data = data[:args.max_samples]
                print(f"Limited dataset to {args.max_samples} samples")
                
            print(f"Successfully processed {len(data)} samples")
            
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            print("Falling back to local file")
            with open(args.data_path, "r") as f:
                data = json.load(f)
    else:
        # Load data from local file
        print(f"Loading dataset from local file: {args.data_path}")
        with open(args.data_path, "r") as f:
            data = json.load(f)
        
        # Limit the number of samples if specified
        if args.max_samples and args.max_samples < len(data):
            data = data[:args.max_samples]
            print(f"Limited dataset to {args.max_samples} samples")

    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    results = None

    for sample in tqdm((data)):
        conv = get_conversation_template("vicuna")
        conv.messages = []
        conv.append_message(conv.roles[0], sample["instruction"])
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        steps = args.steps
        logits_ids = []
        medusa_topk_ids = []

        with torch.inference_mode():
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()
            model.current_length_data.zero_()  # this is for rerun
            reset_medusa_mode(model)
            medusa_logits, outputs, logits = model(
                input_ids, past_key_values=past_key_values, output_orig=True, medusa_forward=True
            )
            _, medusa_topk = medusa_logits[...,-1,:].topk(20, dim=-1)
            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            medusa_topk_ids.append(medusa_topk.detach().cpu())
            for _ in range(steps):
                medusa_logits, outputs, logits = model(
                    input_id, past_key_values=past_key_values, output_orig=True, medusa_forward=True
                )
                _, medusa_topk = medusa_logits[...,-1,:].topk(20, dim=-1)
                input_id = logits[:, -1:].argmax(dim=-1)
                logits_ids.append(input_id.detach().cpu())
                medusa_topk_ids.append(medusa_topk.detach().cpu())
            logits_ids = torch.stack(logits_ids, dim=0)
            medusa_topk_ids = torch.stack(medusa_topk_ids, dim=0).squeeze(2)
            if results is None:
                results = get_accuracies(medusa_topk_ids, logits_ids)
            else:
                # cat sub results
                cur_results = get_accuracies(medusa_topk_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    save_path = os.path.join(args.save_dir, f"{args.model_name}_heads{args.medusa_num_heads}_accuracy.pt")
    torch.save(results, save_path)
    
    # Calculate and print the accuracy results
    print(f"\nAccuracy results for {args.medusa_num_heads} Medusa heads:")
    for i in range(len(results)):
        accuracy = results[i].float().mean().item() * 100
        print(f"  Head {i+1}: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medusa Model Evaluator")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained Medusa model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--medusa_num_heads", type=int, default=6,
                        help="Number of medusa heads to use for evaluation (must be <= total heads in the model).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the evaluation data in JSON format (used when use_huggingface=False).")
    parser.add_argument("--use_huggingface", action="store_true",
                        help="Whether to load the dataset from HuggingFace instead of local file.")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca_eval",
                        help="Name of the dataset on HuggingFace (used when use_huggingface=True).")
    parser.add_argument("--save_dir", type=str, default="./accuracy_data",
                        help="Directory to save the results.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to run the model.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for quicker testing).")
    args = parser.parse_args()

    # If the save directory doesn't exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Check if we have a data path or are using HuggingFace
    if not args.use_huggingface and not args.data_path:
        parser.error("Either --data_path must be provided or --use_huggingface must be set")
        
    main(args)