from datasets import load_dataset
from prompts import GSM8K, GPQA, GSM_Hard, MATH, MMLU, AIME

import os
import json
import re
import random
from collections import Counter


base_path = f"xxx/xxx/.../rethinking_prompting"
log_path = os.path.join(base_path, "logs")


def get_cost(model_name, prompt_tokens, completion_tokens):
    prompt_tokens = float(prompt_tokens)
    completion_tokens = float(completion_tokens)
    if "gemini" in model_name:
        cost = prompt_tokens * 0.075 + completion_tokens * 0.3
    elif model_name == "gpt-3.5-turbo-0613":
        cost = prompt_tokens * 1.5 + completion_tokens * 2
    elif model_name == "gpt-4o-mini":
        cost = prompt_tokens * 0.15 + completion_tokens * 0.6
    else:
        cost = prompt_tokens * 0.15 + completion_tokens * 0.6
    cost = cost / 10 ** 6
    return cost


def find_most_common_elements(outputs):
    outputs = [output for output in outputs if output != None]
    if outputs == []:
        return None
    counter = Counter(outputs)
    max_count = max(counter.values())
    most_common_elements = [element for element, count in counter.items() if count == max_count]
    return most_common_elements, max_count


def get_unique_most_common_answer(outputs):
    outputs = [output for output in outputs if output != None]
    if outputs == []:
        return None
    most_common_elements, max_count = find_most_common_elements(outputs)
    most_common_answer = random.choice(most_common_elements)
    return most_common_answer


def load_GPQA_examples(dataset_list, seed: int):
    random.seed(seed)

    def shuffle_choices_and_create_example(row):
        list_choices = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"], row["Correct Answer"]]
        random.shuffle(list_choices)
        
        example = {}
        example["problem"] = row["Question"]
        example["subject"] = row["High-level domain"]
        example["choices"] = list_choices
        example["answer"] = chr(list_choices.index(row["Correct Answer"]) + 65)
        return example

    return [shuffle_choices_and_create_example(row) for row in dataset_list]


def read_dataset(args):
    dataset = args.dataset
    if dataset == "GSM8K":
        ds = load_dataset("openai/gsm8k", "main")[args.split]
        dataset_list = [d for d in ds]
    elif dataset == "GSM-Hard":
        ds = load_dataset("reasoning-machines/gsm-hard")
        dataset_list = [d for d in ds["train"]]
    elif dataset == "MATH":
        ds = load_dataset("HuggingFaceH4/MATH-500")[args.split]
        dataset_list = [d for d in ds]
    elif dataset == "GPQA":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
        dataset_list = load_GPQA_examples([d for d in ds], args.seed)
    elif "MMLU" in dataset:
        subject = dataset.split("-")[-1]
        dataset_list = load_dataset("cais/mmlu", subject)[args.split]
    elif dataset == "AIME_2024":
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from modelscope.msdatasets import MsDataset
        ds =  MsDataset.load("AI-ModelScope/AIME_2024", subset_name="default", split="train")
        dataset_list = [d for d in ds]
    return dataset_list
        
        
def examine_output(dataset, output, key):
    if dataset in ["GSM8K"]:
        if output != None and abs(float(re.sub(r"[^0-9.-]", "", str(key))) - output) < 10**(-4):
            return True
    elif dataset in ["GSM-Hard"]:
        if output != None and abs(float(key) - output) < 10**(-4):
            return True
    elif dataset in ["MATH", "AIME_2024"]:
        if is_equiv(str(key), output):
            return True
    elif dataset in ["GPQA", "CommonSenseQA"] or "MMLU" in dataset:
        if key == output:
            return True
        return False


def record_logs(logs, args):
    if not os.path.exists(os.path.join(log_path, args.dataset, args.model_name)):
        os.makedirs(os.path.join(log_path, args.dataset, args.model_name))
    path = os.path.join(log_path, args.dataset, args.model_name, f"{args.reasoning}_{args.shot}_{args.n}.json")
    with open(path, "w") as f:
        json.dump(logs, f, indent = 4)
        
        
def record_a_logs(logs, args):
    if not os.path.exists(os.path.join(log_path, args.dataset)):
        os.mkdir(os.path.join(log_path, args.dataset))
    if not os.path.exists(os.path.join(log_path, args.dataset, args.model_name)):
        os.mkdir(os.path.join(log_path, args.dataset, args.model_name))
    path = os.path.join(log_path, args.dataset, args.model_name, f"{args.reasoning}_{args.shot}_{args.n}.json")
    with open(path, "a") as f:
        json.dump(logs, f, indent = 4)
        
        
def read_logs(args):
    path = os.path.join(log_path, args.dataset, args.model_name, f"{args.reasoning}_{args.shot}_{args.n}.json")
    if os.path.exists(path):
        with open (path, "r") as f:
            logs = json.loads(f.read())
    else:
        logs = []
    return logs


def construct_message(args, agents, question, idx):

    prefix_string = "These are the answers to the question from other agents: "

    for agent in agents:
        agent_response = agent[idx]
        response = "\n\n One agent answer: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if args.dataset in ["GSM8K", "GSM-Hard"]:
        prefix_string = prefix_string + "\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}.".format(question) + " Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response."
    elif args.dataset in ["GPQA"]:
        prefix_string = prefix_string + "\n\n Using the solutions from other agents as additional information, can you provide your answer to the problem? \n The original problem is {}. ".format(question) + GPQA.prompt_format
    elif "MMLU" in args.dataset:
        prefix_string = prefix_string + "\n\n Using the solutions from other agents as additional information, can you provide your answer to the problem? \n The original problem is {}. ".format(question) + MMLU.prompt_format
    elif args.dataset in ["MATH", "AIME_2024"]:
        prefix_string = prefix_string + "\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. ".format(question) + MATH.prompt_format
    else:
        raise ValueError(f"{args.dataset} is not included in MAD!")
    
    return prefix_string


def create_prompt(args, index = None):
    prompts = []
    if index != None:
        range_ = [index]
    else:
        range_ = range(0, len(args.questions))
        
    for i in range_:
        args.question = args.questions[i]
        if args.dataset == "GSM8K":
            if args.reasoning == "DiP":
                prompt = GSM8K.io.replace("{question}", args.question)
            elif args.reasoning == "CoT":
                assert args.shot in [0, 1, 5]
                if args.shot == 0:
                    prompt = GSM8K.cot_0_shot.replace("{question}", args.question)
                elif args.shot == 1:
                    prompt = GSM8K.cot_1_shot.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = GSM8K.cot_5_shot.replace("{question}", args.question) 
            elif args.reasoning == "L2M":
                if args.shot == 1:
                    prompt = GSM8K.Least_to_Most_1_shot.replace("{question}", args.question) 
            elif args.reasoning == "ToT":
                if args.shot == 3:
                    prompt = GSM8K.tot_3_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"]) + GSM8K.tot_post
                elif args.shot == 5:
                    prompt = GSM8K.tot_5_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"]) + GSM8K.tot_post
                elif args.shot == 10:
                    prompt = GSM8K.tot_10_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"], solution6 = args.records_tot[i][5]["output"], solution7 = args.records_tot[i][6]["output"], solution8 = args.records_tot[i][7]["output"], solution9 = args.records_tot[i][8]["output"], solution10 = args.records_tot[i][9]["output"]) + GSM8K.tot_post
            elif args.reasoning == "AnP":
                if args.shot == 1:
                    prompt = GSM8K.anologous_1_prompt.replace("{question}", args.question) 
                elif args.shot == 3:
                    prompt = GSM8K.anologous_3_prompt.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = GSM8K.anologous_5_prompt.replace("{question}", args.question)
            elif args.reasoning == "SBP":
                if "principles" not in args:
                    prompt = GSM8K.SBP_extract.replace("{question}", args.question) 
                else:
                    prompt = GSM8K.SBP_answer.replace("{question}", args.question)
                    prompt = prompt.replace("{principles}", args.principles) 
                    
        elif args.dataset == "GSM-Hard":
            if args.reasoning == "DiP":
                prompt = GSM_Hard.io.replace("{question}", args.question)
            elif args.reasoning == "CoT":
                assert args.shot in [0, 1, 5]
                if args.shot == 0:
                    prompt = GSM_Hard.cot_0_shot.replace("{question}", args.question)
                elif args.shot == 1:
                    prompt = GSM_Hard.cot_1_shot.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = GSM_Hard.cot_5_shot.replace("{question}", args.question) 
            elif args.reasoning == "L2M":
                if args.shot == 1:
                    prompt = GSM_Hard.Least_to_Most_1_shot.replace("{question}", args.question) 
            elif args.reasoning == "ToT":
                if args.shot == 3:
                    prompt = GSM_Hard.tot_3_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"]) + GSM_Hard.tot_post
                elif args.shot == 5:
                    prompt = GSM_Hard.tot_5_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"]) + GSM_Hard.tot_post
                elif args.shot == 10:
                    prompt = GSM_Hard.tot_10_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"], solution6 = args.records_tot[i][5]["output"], solution7 = args.records_tot[i][6]["output"], solution8 = args.records_tot[i][7]["output"], solution9 = args.records_tot[i][8]["output"], solution10 = args.records_tot[i][9]["output"]) + GSM_Hard.tot_post
            elif args.reasoning == "AnP":
                if args.shot == 1:
                    prompt = GSM_Hard.anologous_1_prompt.replace("{question}", args.question) 
                elif args.shot == 3:
                    prompt = GSM_Hard.anologous_3_prompt.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = GSM_Hard.anologous_5_prompt.replace("{question}", args.question)
            elif args.reasoning == "SBP":
                if "principles" not in args:
                    prompt = GSM_Hard.SBP_extract.replace("{question}", args.question) 
                else:
                    prompt = GSM_Hard.SBP_answer.replace("{question}", args.question)
                    prompt = prompt.replace("{principles}", args.principles) 
                    
        elif args.dataset == "GPQA":
            args.choices = args.choiceses[i]
            if args.reasoning == "DiP":
                prompt = GPQA.io.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "CoT":
                assert args.shot in [0, 1, 5]
                if args.shot == 0:
                    prompt = GPQA.cot_0_shot.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                elif args.shot == 1:
                    prompt = GPQA.cot_1_shot.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3]) 
                elif args.shot == 5:
                    prompt = GPQA.cot_5_shot.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3]) 
            elif args.reasoning == "L2M":
                if args.shot == 0:
                    prompt = GPQA.Least_to_Most_0_shot.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "ToT":
                if args.shot == 3:
                    prompt = GPQA.tot_3_solutions.format(question=args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"]) + GPQA.tot_post
                elif args.shot == 5:
                    prompt = GPQA.tot_5_solutions.format(question=args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"]) + GPQA.tot_post
                elif args.shot == 10:
                    prompt = GPQA.tot_10_solutions.format(question=args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"], solution6 = args.records_tot[i][5]["output"], solution7 = args.records_tot[i][6]["output"], solution8 = args.records_tot[i][7]["output"], solution9 = args.records_tot[i][8]["output"], solution10 = args.records_tot[i][9]["output"]) + GPQA.tot_post
            elif args.reasoning == "AnP":
                if args.shot == 1:
                    prompt = GPQA.anologous_1_prompt.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                elif args.shot == 3:
                    prompt = GPQA.anologous_3_prompt.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                elif args.shot == 5:
                    prompt = GPQA.anologous_5_prompt.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "SBP":
                if "principles" not in args:
                    prompt = GPQA.SBP_extract.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                else:
                    prompt = GPQA.SBP_answer.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], principles = args.principles)
                    
        elif "MMLU" in args.dataset:
            args.choices = args.choiceses[i]
            if args.reasoning == "DiP":
                prompt = MMLU.io.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "CoT":
                assert args.shot in [0, 1, 5]
                if args.shot == 0:
                    prompt = MMLU.cot_0_shot.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "L2M":
                if args.shot == 0:
                    prompt = MMLU.Least_to_Most_0_shot.format(question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "ToT":
                if args.shot == 3:
                    prompt = MMLU.tot_3_solutions.format(question=args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"]) + MMLU.tot_post
                elif args.shot == 5:
                    prompt = MMLU.tot_5_solutions.format(question=args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"]) + MMLU.tot_post
                elif args.shot == 10:
                    prompt = MMLU.tot_10_solutions.format(question=args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"], solution6 = args.records_tot[i][5]["output"], solution7 = args.records_tot[i][6]["output"], solution8 = args.records_tot[i][7]["output"], solution9 = args.records_tot[i][8]["output"], solution10 = args.records_tot[i][9]["output"]) + MMLU.tot_post
            elif args.reasoning == "AnP":
                if args.shot == 1:
                    prompt = MMLU.anologous_1_prompt.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                elif args.shot == 3:
                    prompt = MMLU.anologous_3_prompt.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                elif args.shot == 5:
                    prompt = MMLU.anologous_5_prompt.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
            elif args.reasoning == "SBP":
                if "principles" not in args:
                    if "physics" in args.subject:
                        prompt = MMLU.SBP_extract_physics.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                    elif "chemistry" in args.subject:
                        prompt = MMLU.SBP_extract_chemistry.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                    else:
                        prompt = MMLU.SBP_extract.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3])
                else:
                    if "physics" in args.subject:
                        prompt = MMLU.SBP_answer_physics.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], principles = args.principles)
                    elif "chemistry" in args.subject:
                        prompt = MMLU.SBP_answer_chemistry.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], principles = args.principles)
                    else:
                        prompt = MMLU.SBP_answer.format(subject = args.subject, question = args.question, choice1 = args.choices[0], choice2 = args.choices[1], choice3 = args.choices[2], choice4 = args.choices[3], principles = args.principles)
                        
        elif args.dataset == "MATH":
            if args.reasoning == "DiP":
                prompt = MATH.io.replace("{question}", args.question)
            elif args.reasoning == "CoT":
                assert args.shot in [0, 1, 5]
                if args.shot == 0:
                    prompt = MATH.cot_0_shot.replace("{question}", args.question)
                elif args.shot == 1:
                    prompt = MATH.cot_1_shot.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = MATH.cot_5_shot.replace("{question}", args.question) 
            elif args.reasoning == "L2M":
                if args.shot == 0:
                    prompt = MATH.Least_to_Most_0_shot.replace("{question}", args.question) 
            elif args.reasoning == "ToT":
                if args.shot == 3:
                    prompt = MATH.tot_3_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"]) + MATH.tot_post
                elif args.shot == 5:
                    prompt = MATH.tot_5_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"]) + MATH.tot_post
                elif args.shot == 10:
                    prompt = MATH.tot_10_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"], solution6 = args.records_tot[i][5]["output"], solution7 = args.records_tot[i][6]["output"], solution8 = args.records_tot[i][7]["output"], solution9 = args.records_tot[i][8]["output"], solution10 = args.records_tot[i][9]["output"]) + MATH.tot_post
            elif args.reasoning == "AnP":
                if args.shot == 1:
                    prompt = MATH.anologous_1_prompt.replace("{question}", args.question) 
                elif args.shot == 3:
                    prompt = MATH.anologous_3_prompt.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = MATH.anologous_5_prompt.replace("{question}", args.question)
            elif args.reasoning == "SBP":
                if "principles" not in args:
                    prompt = MATH.SBP_extract.replace("{question}", args.question) 
                else:
                    prompt = MATH.SBP_answer.replace("{question}", args.question)
                    prompt = prompt.replace("{principles}", args.principles) 
                    
        elif args.dataset == "AIME_2024":
            if args.reasoning == "DiP":
                prompt = AIME.io.replace("{question}", args.question)
            elif args.reasoning == "CoT":
                assert args.shot in [0, 1, 5]
                if args.shot == 0:
                    prompt = AIME.cot_0_shot.replace("{question}", args.question) 
            elif args.reasoning == "L2M":
                if args.shot == 0:
                    prompt = AIME.Least_to_Most_0_shot.replace("{question}", args.question) 
            elif args.reasoning == "ToT":
                if args.shot == 3:
                    prompt = AIME.tot_3_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"]) + AIME.tot_post
                elif args.shot == 5:
                    prompt = AIME.tot_5_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"]) + AIME.tot_post
                elif args.shot == 10:
                    prompt = AIME.tot_10_solutions.format(question=args.question, solution1 = args.records_tot[i][0]["output"], solution2 = args.records_tot[i][1]["output"], solution3 = args.records_tot[i][2]["output"], solution4 = args.records_tot[i][3]["output"], solution5 = args.records_tot[i][4]["output"], solution6 = args.records_tot[i][5]["output"], solution7 = args.records_tot[i][6]["output"], solution8 = args.records_tot[i][7]["output"], solution9 = args.records_tot[i][8]["output"], solution10 = args.records_tot[i][9]["output"]) + AIME.tot_post
            elif args.reasoning == "AnP":
                if args.shot == 1:
                    prompt = AIME.anologous_1_prompt.replace("{question}", args.question) 
                elif args.shot == 3:
                    prompt = AIME.anologous_3_prompt.replace("{question}", args.question) 
                elif args.shot == 5:
                    prompt = AIME.anologous_5_prompt.replace("{question}", args.question)
            elif args.reasoning == "SBP":
                if "principles" not in args:
                    prompt = AIME.SBP_extract.replace("{question}", args.question) 
                else:
                    prompt = AIME.SBP_answer.replace("{question}", args.question)
                    prompt = prompt.replace("{principles}", args.principles) 
        else:
            raise ValueError(f"{args.dataset} is not included in def create_prompt!")
        prompts.append(prompt)
        
    return prompts


def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    # 6+9j -> 6+9i
    string = string.replace("j", "i")
    # if empty, return empty string
    
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def replace_sqrt_with_power(s):
    pattern = r"\\sqrt\{([^}]+)\}"
    return re.sub(pattern, r"(\1)**0.5", s)


def replace_pi(s):
    # result = re.sub(r"(?<!\S)pi(?!\S)", "3.1415926", s)
    result = re.sub(r"\\pi", "*3.1415926", s)
    result = re.sub(r"pi", "3.1415926", result)
    return result


def replace_with_asterisk(s):
    result = re.sub(r"(\d)(pi|\\sqrt|\\frac)", r"\1*\2", s)
    return result


def replace_frac(s):
    result = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    result = re.sub(r"frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", result)
    return result


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

    
def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None 
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval
    
    
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def parse_answer(args, input_str):
    solution = None
    if args.dataset in ["GSM8K", "GSM-Hard"]:
        pattern = r"boxed\{(.*?)\}"
        matches = re.findall(pattern, input_str)

        for match_str in matches[::-1]:
            match_str = match_str.split("=")[-1]
            if "boxed" not in match_str:
                solution = re.sub(r"[^0-9.-]", "", match_str)
            else:
                solution = parse_answer(args, match_str)
            if solution:
                break
        
        if solution == None or solution == "":
            pattern = r"boxed\{(.*)\}"
            matches = re.findall(pattern, input_str)

            for match_str in matches[::-1]:
                if "boxed" not in match_str:
                    solution = re.sub(r"[^0-9.-]", "", match_str)
                else:
                    solution = parse_answer(args, match_str)
                if solution:
                        break

        if solution == None or solution == "":
            pattern = r"\{([0-9 \-.,$]*)\}"
            matches = re.findall(pattern, input_str)

            for match_str in matches[::-1]:
                solution = re.sub(r"[^0-9.-]", "", match_str)
                if solution:
                    break
                
        if solution == None or solution == "":
            pattern = r"\*\*(.*)\*\*"
            matches = re.findall(pattern, input_str)
        
            for match_str in matches[::-1]:
                solution = re.sub(r"[^0-9.-]", "", match_str)
                if solution:
                    break
                
        if solution == None or solution == "":
            matches = re.findall(r"[0-9\-.,$]+", input_str)
            for match_str in matches[::-1]:
                if re.findall(r"\d+", match_str) != []:
                    solution =  re.sub(r"[^0-9.-]", "", match_str)
                    if solution[-1] == ".":
                        solution = solution[:-1]
                    break
        try:
            solution = float(solution)
        except:
            solution = None
    elif args.dataset in ["GPQA"] or "MMLU" in args.dataset:
        answers = re.findall(r"correct answer is \*\*(.*)\*\*", input_str)
        if args.dataset in ["GPQA"] or "MMLU" in args.dataset:
            letters = ["A", "B", "C", "D"]
        for answer in answers[::-1]:
            if answer[0] not in letters:
                try:
                    solution = re.search(r"\((.)\)", answer).group(1)[0]
                    if solution in letters:
                        return solution
                    else:
                        return None
                except:
                    solution = "M"
            else:
                solution = answer[0]
                return solution
        
        answers = re.findall(r"correct answer is (.?)", input_str)
        for answer in answers[::-1]:
            if answer[0] not in letters:
                try:
                    solution = re.search(r"correct answer is \((.?)\)", input_str).group(1)[0]
                    return solution
                except:
                    answer = "M"
            else:
                solution = answer[0]
                return solution
        answers = re.findall(r"\((.)\)", input_str)
        for answer in answers[::-1]:
            if answer[0] in letters:
                solution = answer[0]
                return solution
        answers = re.findall(r"\{(.)\}", input_str)
        for answer in answers[::-1]:
            if answer[0] in letters:
                solution = answer[0]
    elif args.dataset in ["MATH", "AIME_2024"]:
        return remove_boxed(last_boxed_only_string(input_str))
    else:
        raise ValueError(f"{args.dataset} is not in def parse_answer!")
    return solution


def parse_best_solution(input_str):
    pattern = r"index of the best solution is (\d+)"
    matches = re.findall(pattern, input_str)
    
    if matches:
        return matches[-1]
    else:
        pattern = r"\*\*(\d+)\*\*"
        matches = re.findall(pattern, input_str)
        
        for match_str in matches[::-1]:
            if match_str:
                return match_str
        return None

    
def parse_best_method(s):
    start_str = "most suitable method is "
    start_index = s.find(start_str)
    if start_index == -1:
        return ""
    start_index += len(start_str)
    end_index = start_index
    while end_index < len(s) and s[end_index] not in ".,!?;:\n":
        end_index += 1
    return s[start_index:end_index].strip()


def check_solution_verdict(output_str):
    pattern = r"solution is (right|wrong)"
    match = re.search(pattern, output_str)
    
    if match:
        return match.group(1)
    else:
        s = "random_" + random.choice(["right", "wrong"])
        return s
