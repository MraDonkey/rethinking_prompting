from dataset import *
from model import load_model, LLM_generate
from dataset import create_prompt

import os
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from huggingface_hub import login

hf_token = "hf_YourTokenHere"  # Replace with your token
login(token=hf_token)


refine1_feeback_prompt = "Review your previous answer and find problems with your answer."
refine1_refine_prompt = "Based on the problems you found, improve your answer."


def concat_refine_records(records, output, string):
    for i in range(0, len(output)):
        for j in range(0, len(output[i])):
            records[i][j][string] = output[i][j]


def concat_refine_messages(messages, output):
    for i in range(0, len(output)):
        if isinstance(output, str):
            messages[i].append(output)
        else:
            messages[i].append(output[i][0]["output"])


def get_model_outputs(args):
    if args.reasoning in ["DiP", "CoT", "AnP", "L2M"]:
        args.query = create_prompt(args)
        if args.verbal:
            print(args.query)
        records = LLM_generate(args)
        
    elif args.reasoning == "S-RF":
        args.reasoning = "DiP"
        queries = create_prompt(args)
        args.reasoning = "S-RF"
        args.messages = [[query] for query in queries]
        origin_output = LLM_generate(args)
        records = []
        for i in range(0, len(origin_output)):
            record_ = []
            for j in range(0, len(origin_output[i])):
                record = {}
                record["output0"] = origin_output[i][j]
                record_.append(record)
            records.append(record_)
        with ThreadPoolExecutor(max_workers=5) as executor:
            parameters = (args.messages, origin_output)
            executor.submit(concat_refine_messages, *parameters)
        for j in tqdm(range(0, args.rounds)):
            with ThreadPoolExecutor(max_workers=5) as executor:
                parameters = (args.messages, refine1_feeback_prompt)
                executor.submit(concat_refine_messages, *parameters)
            output = LLM_generate(args)
            if args.verbal:
                print(output)
            with ThreadPoolExecutor(max_workers=5) as executor:
                parameters = (records, output, f"problems{j+1}")
                executor.submit(concat_refine_records, *parameters)
                parameters = (args.messages, output)
                executor.submit(concat_refine_messages, *parameters)
                parameters = (args.messages, refine1_refine_prompt + " " + PROMPT_FORMAT)
                executor.submit(concat_refine_messages, *parameters)
            output = LLM_generate(args)
            if args.verbal:
                print(output)
            with ThreadPoolExecutor(max_workers=5) as executor:
                parameters = (records, output, f"output{j+1}")
                executor.submit(concat_refine_records, *parameters)
                parameters = (args.messages, output)
                executor.submit(concat_refine_messages, *parameters)
                
    elif args.reasoning == "ToT":
        records = []
        args.query = create_prompt(args)
        if args.verbal:
            print(args.query)
        l = len(args.questions)
        output_choices = LLM_generate(args)
        for i in range(0, l):
            record_ = []
            record = {}
            record["solutions"] = args.records_tot[i]
            for j in range(0, args.num):
                record["choose"] = output_choices[i][j]
                record_.append(record)
            records.append(record_)
            
    elif args.reasoning == "SBP":
        records = []
        args.query = create_prompt(args)
        if args.verbal:
            print(args.query)
        principles = LLM_generate(args)
        num = args.num
        l = len(args.questions)
        args.num = 1
        args.query = []
        for i in range(0, l):
            for j in range(0, num):
                record = {}
                record["principles"] = principles[i][j]
                args.principles = record["principles"]["output"]
                args.query.append(create_prompt(args, i)[0])
        del args.principles
        solutions = LLM_generate(args)
        for i in range(0, l):
            record_ = []
            for j in range(0, num):
                record = {}
                record["principles"] = principles[i][j]
                record["solution"] = solutions[num * i + j][0]
                record_.append(record)
            records.append(record_)
        args.num = num
        
    elif args.reasoning == "MAD":
        records = {}
        args.reasoning = "DiP"
        agent_contexts = [create_prompt(args) for agent in range(0, 3)]
        args.reasoning = "MAD"
        if args.verbal:
            print(args.query)
        if args.continue_:
            for round in range(args.rounds):
                if round < len(args.records):
                    outputs = [output["output"] for output in args.records[f"round{round+1}"]]
                    records[f"round{round+1}"] = args.records[f"round{round+1}"]
                else:
                    records[f"round{round+1}"] = []
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(args, agent_contexts_other, args.question, 2*round - 1)
                        agent_context.append(message)
                        args.messages = [agent_context]
                        if round < len(args.records):
                            assistant_message = outputs[i]
                        else:
                            record = LLM_generate(args)[0][0]
                            records[f"round{round+1}"].append(record)
                            assistant_message = record["output"]
                    else:
                        assistant_message = outputs[i]
                    agent_context.append(assistant_message)
        else:
            for round in range(args.rounds):
                records[f"round{round+1}"] = []
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(args, agent_contexts_other, args.question, 2*round - 1)
                        agent_context.append(message)
                    args.messages = [agent_context]
                    record = LLM_generate(args)[0][0]
                    records[f"round{round+1}"].append(record)
                    assistant_message = record["output"]
                    agent_context.append(assistant_message)
        records = [[records]]
        
    return records


def handle_tot_reasoning(args):
    """Handle Tree of Thoughts (ToT) reasoning setup."""
    logs_tot = []
    shot = args.shot
    args.shot = 0
    args.reasoning = "CoT"
    
    for j in range(shot):
        args.n = j
        logs_tot.append(read_logs(args))
        
    args.reasoning = "ToT"
    args.shot = shot
    return logs_tot


def setup_mad_reasoning(args):
    """Setup for MAD (Multi-Agent Debate) reasoning, reusing the results of DiP."""
    reasoning = args.reasoning
    args.reasoning = "DiP"
    
    # Collect initial logs
    args.n = 0
    logs_DiP_0 = read_logs(args)
    args.n = 1
    logs_DiP_1 = read_logs(args)
    args.n = 2
    logs_DiP_2 = read_logs(args)
    
    assert len(logs_DiP_0) == len(logs_DiP_1) == len(logs_DiP_2), "Logs of DiP for MAD reasoning must be of the same length."
    assert len(logs_DiP_0) > 0, "To use MAD reasoning, DiP logs must be available."
    args.reasoning = reasoning
    return logs_DiP_0, logs_DiP_1, logs_DiP_2


if __name__ == "__main__":
    # Model and dataset configurations
    MODEL_CONFIGS = {
        "Qwen-2.5": "Qwen/Qwen2.5-7B-Instruct",
        "Llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "GLM-4": "THUDM/glm-4-9b-chat",
        "Gemini": "gemini-1.5-flash",
        "GPT4-mini": "gpt-4o-mini",
        "Phi-3.5": "microsoft/Phi-3.5-mini-instruct"
    }
    
    DATASET_CONFIGS = {
        "math": ["GSM8K", "GSM-Hard", "MATH", "AIME_2024"],
        "science": [
            "MMLU-high_school_physics",
            "MMLU-high_school_chemistry",
            "MMLU-high_school_biology",
            "GPQA",
        ],
    }
    
    PROMPT_FORMATS = {
        "GSM8K": GSM8K.prompt_format,
        "GPQA": GPQA.prompt_format,
        "GSM-Hard": GSM_Hard.prompt_format,
        "MATH": MATH.prompt_format,
        "MMLU-high_school_physics": MMLU.prompt_format,
        "MMLU-high_school_chemistry": MMLU.prompt_format,
        "MMLU-high_school_biology": MMLU.prompt_format,
        "AIME_2024": AIME.prompt_format
    }
    
    # Set up argument parser with improved descriptions
    parser = argparse.ArgumentParser(description="Large Language Model Evaluation Framework")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default=MODEL_CONFIGS["Qwen-2.5"],
                      choices=list(MODEL_CONFIGS.values()))
    parser.add_argument("--model_type", type=str, default="vllm",
                      choices=["vllm", "gemini", "openai"])
    
    # Dataset configuration                  
    parser.add_argument("--dataset", type=str, default="GSM8K",
                      choices=[ds for group in DATASET_CONFIGS.values() for ds in group])
    parser.add_argument("--split", type=str, default="test")
    
    # Reasoning strategy configuration
    parser.add_argument("--reasoning", type=str, default="DiP",
                      choices=["DiP", "CoT", "L2M", "SBP", "AnP", "S-RF", "ToT", "MAD"],
                      help="Reasoning prompting strategy")
    parser.add_argument("--shot", type=int, default=0,
                      help='''DiP, SBP: Shot is fixed at 0, while SBP using 1-shot on MMLU.
                      CoT, L2M: Number of examples in few-shot prompting. 
                      AnP: Number of analogous problems to generate.
                      S-RF: Shot is fixed at 0.
                      ToT: Number of reasoning paths to generate.
                      MAD: Shot is fixed at 0, using 3 agents for debate.
                      ''')
    
    # Processing configuration
    parser.add_argument("--max_num_workers", type=int, default=1,
                      help="Maximum number of workers for parallel processing for API-based models")
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    
    # Generation range configuration 
    parser.add_argument("--range_begin", type=int, default=0)
    parser.add_argument("--range_end", type=int, default=16)
    
    # Hardware configuration
    parser.add_argument("--gpu", type=str, default="4,5", help="GPU IDs to use, e.g., '0,1'")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                      help="Data type for VLLM")
    
    # API configuration
    parser.add_argument("--google_api_key", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_base_url", type=str)
    
    # Other options
    parser.add_argument("--verbal", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    # Set up basic configurations
    args.range = range(args.range_begin, args.range_end)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Initialize prompt format based on dataset
    PROMPT_FORMAT = PROMPT_FORMATS.get(args.dataset)
    
    # Reset messaging configuration
    args.system = None
    args.messages = None
    args.query = None

    # Special handling for MAD reasoning
    if args.reasoning == "MAD":
        args.batchsize = 1
        args.max_num_workers = 1
        
    # Load dataset and prepare model
    dataset_list = read_dataset(args)
    model_name = args.model_name
    args.model_name = model_name.split("/")[-1]

    # Special handling for different reasoning strategies
    if args.reasoning in ["DiP", "SBP"]:
        args.shot = 0
    elif args.reasoning in ["CoT"]:
        if args.dataset in ["GSM8K", "GSM-Hard", "MATH"]:
            assert args.shot in [0, 1, 5], "CoT reasoning requires 0, 1, or 5 shots for GSM8K, GSM-Hard, and MATH datasets"
        else:
            assert args.shot == 0, f"CoT reasoning requires 0-shot prompting for {args.dataset} dataset"
    elif args.reasoning == "AnP":
        assert args.shot in [1, 3, 5], "AnP reasoning requires 1, 3, or 5 shots, i.e., generating 1, 3, or 5 analogous problems"
    elif args.reasoning in ["S-RF", "MAD"]:
        args.shot = 0
        assert len(args.range) == 1, "Range from the beginning to the end must be 1 for S-RF and MAD reasoning"
    elif args.reasoning == "ToT":
        assert args.shot in [3, 5, 10], "ToT reasoning requires 3, 5, or 10 shots, i.e., generating 3, 5, or 10 reasoning paths"
        logs_tot = handle_tot_reasoning(args)
        
    args.num = len(args.range)
    
    # Handle MAD reasoning specific setup
    if args.reasoning == "MAD":
        logs_DiP_0, logs_DiP_1, logs_DiP_2 = setup_mad_reasoning(args)
        
    print(f"{'='*10}{args.dataset}===={args.reasoning}===={args.shot}{'='*32}")
    
    # Process logs and validate progress
    logs_all = []
    for j in args.range:
        args.n = j
        logs_all.append(read_logs(args))
    args.n = args.range[0]
    logs = logs_all[0]
    begin_num = len(logs)
    
    assert begin_num < len(dataset_list), "Processing already completed. Please check logs."
    
    # Initialize model
    args.model_name = model_name
    load_model(args)
    args.model_name = model_name.split("/")[-1]
    
    
    letters = ["A", "B", "C", "D", "E"]
    if args.reasoning == "MAD":
        if len(logs) == len(dataset_list):
            rounds = len(logs[0]["record"])
            if rounds < args.rounds:
                begin_num = 0
                args.continue_ = True
            else:
                args.continue_ = False
        else:
            args.continue_ = False
    
    if args.model_type == "vllm":
        for ii in tqdm(range(begin_num, len(dataset_list), args.batchsize)):
            examples = []
            args.questions = []
            args.choiceses = []
            args.subjects = []
            range_ = range(ii, min(len(dataset_list), ii + args.batchsize))
            args.records_tot = []
            for i in range_:
                if args.reasoning == "ToT":
                    records = []
                    for j in range(0, args.shot):
                        records.append(logs_tot[j][i]["record"])
                    args.records_tot.append(records)
                example = copy.deepcopy(dataset_list[i])
                if args.dataset == "GSM8K":
                    args.questions.append(example["question"])
                    answer = example["answer"]
                    key = answer.split("#### ")[-1]
                    example["key"] = key
                elif args.dataset == "GPQA":
                    args.questions.append(example['problem'])
                    args.choiceses.append(example['choices'])
                    args.subjects.append(example["subject"].lower())
                    example["question"] = example.pop("problem")
                    example["key"] = example["answer"]
                elif args.dataset == "GSM-Hard":
                    args.questions.append(example["input"])
                    example["question"] = example.pop("input")
                    example["key"] = example.pop("target")      
                elif args.dataset == "MATH":
                    args.questions.append(example["problem"])
                    example["question"] = example.pop("problem")
                    example["key"] = example.pop("answer")  
                elif "MMLU" in args.dataset:
                    args.questions.append(example["question"]) 
                    args.subjects.append(example["subject"])
                    args.choiceses.append(example["choices"])
                    example["key"] = letters[example.pop("answer")]     
                elif args.dataset == "AIME_2024":
                    args.questions.append(example["Problem"])
                    example["question"] = example.pop("Problem")
                    example["key"] = example.pop("Answer")
                example["num"] = i
                examples.append(example)
            
            if args.reasoning == "MAD":
                args.previous_record = [logs_DiP_0[i]["record"], logs_DiP_1[i]["record"], logs_DiP_2[i]["record"]]
                if args.continue_:
                    args.records = logs[i]["record"]
                
            records_all = get_model_outputs(args)
                
            for i in range(0, len(range_)):
                records = records_all[i]
                for j, k in enumerate(args.range):
                    new_example = examples[i].copy()
                    new_example["record"] = records[j]
                    logs_all[j].append(new_example)
                    del new_example
            del examples

            for j, k in enumerate(args.range):
                args.n = k
                record_logs(logs_all[j], args)
    else:
        nums = [log["num"] for log in logs]
        remain_data = [i for i in range(len(dataset_list)) if i not in nums]
        remain_ranges = [remain_data[i:i+args.batchsize] for i in range(0, len(remain_data), args.batchsize)]
        for range_ in tqdm(remain_ranges):
            examples = []
            args.questions = []
            args.choiceses = []
            args.subjects = []
            args.records_tot = []
            for i in range_:
                if "tot" in args.reasoning:
                    records = []
                    for j in range(0, args.shot):
                        records.append(logs_tot[j][i]["record"])
                    args.records_tot.append(records)
                example = copy.deepcopy(dataset_list[i])
                if args.dataset == "GSM8K":
                    args.questions.append(example["question"])
                    answer = example["answer"]
                    key = answer.split("#### ")[-1]
                    example["key"] = key
                elif args.dataset == "GPQA":
                    args.questions.append(example['problem'])
                    args.choiceses.append(example['choices'])
                    args.subjects.append(example["subject"].lower())
                    example["key"] = example["answer"]
                elif args.dataset == "GSM-Hard":
                    args.questions.append(example["input"])
                    example["question"] = example.pop("input")
                    example["key"] = example.pop("target")      
                elif args.dataset == "MATH":
                    args.questions.append(example["problem"])
                    example["question"] = example.pop("problem")
                    example["key"] = example.pop("answer")  
                elif "MMLU" in args.dataset:
                    args.questions.append(example["question"]) 
                    args.subjects.append(example["subject"])
                    args.choiceses.append(example["choices"])
                    example["key"] = letters[example.pop("answer")]     
                elif args.dataset == "AIME_2024":
                    args.questions.append(example["Problem"])
                    example["question"] = example.pop("Problem")
                    example["key"] = example.pop("Answer")
                example["num"] = i
                examples.append(example)
            
            if args.reasoning == "MAD":
                args.previous_record = [logs_DiP_0[i]["record"], logs_DiP_1[i]["record"], logs_DiP_2[i]["record"]]
                if args.continue_:
                    args.records = logs[i]["record"]
                
            records_all = get_model_outputs(args)
                
            for i in range(0, len(range_)):
                records = records_all[i]
                for j, k in enumerate(args.range):
                    new_example = examples[i].copy()
                    new_example["record"] = records[j]
                    logs_all[j].append(new_example)
                    del new_example
            del examples

            for j, k in enumerate(args.range):
                args.n = k
                record_logs(logs_all[j], args)
