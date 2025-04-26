"""Generate answers with local models using adaptive Medusa decoding.

Usage:
python3 gen_model_answer_adaptive.py --model-path FasterDecoding/medusa-vicuna-7b-v1.3 --model-id medusa-vicuna-7b-v1.3-adaptive
"""
import argparse
import json
import os
import time
import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template

# Medusa imports
from medusa.model.medusa_model_adaptive import MedusaModel
from medusa.model.medusa_choices import *

def medusa_forward(input_ids, model, tokenizer, medusa_choices, temperature, posterior_threshold, posterior_alpha, top_p=0.8, sampling='typical', fast=True, max_steps=512):
    """
    Forward pass using Medusa's adaptive decoding strategy.
    Uses the new adaptive medusa_generate method which dynamically switches between different head counts.
    
    Args:
        input_ids: Input token IDs
        model: Medusa model
        tokenizer: Tokenizer for the model
        medusa_choices: Tree structures for Medusa decoding
        temperature: Temperature for sampling
        posterior_threshold: Threshold for posterior validation
        posterior_alpha: Another threshold parameter
        top_p: Top-p for nucleus sampling
        sampling: Sampling strategy ('typical' or 'nucleus')
        fast: Whether to use fast decoding
        max_steps: Maximum number of decoding steps
        
    Returns:
        Tuple of (output_ids, new_token_count, steps_taken)
    """
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    
    # Clone input_ids to avoid modifying the original
    input_ids = input_ids.clone()
    input_len = input_ids.shape[1]
    
    # Use the adaptive medusa_generate method
    generation_iter = model.medusa_generate(
        input_ids=input_ids,
        attention_mask=None,
        temperature=temperature,
        max_steps=max_steps,
        medusa_choices=medusa_choices,
        posterior_threshold=posterior_threshold,
        posterior_alpha=posterior_alpha,
        top_p=top_p,
        sampling=sampling,
        fast=fast
    )
    
    # Track generation progress
    steps = 0
    last_text = ""
    
    # Run the generation
    for step_idx, output in enumerate(generation_iter):
        steps = step_idx
        last_text = output["text"]
        
        # Check for EOS token in the text (optional, the generator should handle this)
        if tokenizer.eos_token in last_text:
            break
        
        # Safety check for max steps
        if step_idx >= max_steps - 1:
            break
    
    # Count how many new tokens were generated
    new_token_count = len(tokenizer.encode(last_text, add_special_tokens=False))
    
    # The input_ids were modified in-place by medusa_generate
    # We need to return the final state of input_ids
    return input_ids, new_token_count, steps

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    top_p,
    sampling,
    fast,
    medusa_choices,
    adaptive=True,  # Added adaptive flag
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                posterior_threshold,
                posterior_alpha,
                sampling,
                top_p,
                fast,
                medusa_choices,
                adaptive,  # Pass adaptive flag
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    sampling,
    top_p,
    fast,
    medusa_choices,
    adaptive=True,  # Added adaptive flag
):
    # Medusa model setup
    num_heads = -1
    for choice in medusa_choices:
        if len(choice) > num_heads:
            num_heads = len(choice)

    model = MedusaModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:', model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    print('Using adaptive Medusa decoding:', adaptive)
    
    question = questions[0]

    # warmup
    print('Starting warmup...')
    for _ in range(3):
        conv = get_conversation_template(model_id)
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # some models may error out when generating long outputs
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = medusa_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    medusa_choices,
                    0.7,  # Fixed warmup temperature
                    posterior_threshold,
                    posterior_alpha,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]) :]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print(e)
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    # Main evaluation loop
    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temp = temperature_config[question["category"]]
        else:
            temp = temperature if temperature > 0 else 0.7

        choices = []
        for i in range(num_choices):
            conv = get_conversation_template(model_id)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # some models may error out when generating long outputs
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, idx = medusa_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        medusa_choices,
                        temp,
                        posterior_threshold,
                        posterior_alpha,
                        top_p=top_p,
                        sampling=sampling,
                        fast=fast,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    
                    output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    print(output)
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            
            choices.append({
                "index": i, 
                "turns": turns, 
                "idxs": idxs, 
                "new_tokens": new_tokens, 
                "wall_time": wall_time
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    # Medusa args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for medusa sampling.",
    )
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for medusa sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="The top-p for medusa sampling.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="typical",
        help="The sampling method for medusa sampling.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Whether to use fast decoding.",
    )
    parser.add_argument(
        "--medusa-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The medusa choices for medusa sampling.",
    )
    # Add new adaptive flag
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Whether to use adaptive Medusa decoding with dynamic head switching.",
    )

    args = parser.parse_args()

    # Include adaptive in the model ID if enabled
    adaptive_tag = "-adaptive" if args.adaptive else ""
    args.model_id = (args.model_id + 
                   "-temperature-" + str(args.temperature) + 
                   "-posterior_threshold-" + str(args.posterior_threshold) + 
                   "-posterior_alpha-" + str(args.posterior_alpha) + 
                   "-top_p-" + str(args.top_p) + 
                   "-sampling-" + args.sampling + 
                   "-fast-" + str(args.fast) + 
                   adaptive_tag)
    
    args.medusa_choices = eval(args.medusa_choices)
    
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.posterior_threshold,
        args.posterior_alpha,
        args.top_p,
        args.sampling,
        args.fast,
        args.medusa_choices,
        args.adaptive,  # Pass the adaptive flag
    )

    reorg_answer_file(answer_file)