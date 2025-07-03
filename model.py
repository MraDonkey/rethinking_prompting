import time
from dataset import parse_best_solution, parse_answer

import google.generativeai as genai  # pip install -q -U google-generativeai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed
import pdb
from tqdm import tqdm


def get_messages(args):
    messages = []
    assert args.messages != None or args.query != None
    if "gemini" not in args.model_name:
        if args.messages != None:  
            roles = ['user', 'assistant']
            for i in range(0, len(args.messages)):
                messages_i = []
                assert len(args.messages[i]) % 2 == 1
                if args.system != None:
                    messages_i.append({"role": "system", "content": args.system})
                for j, message in enumerate(args.messages[i]):
                    messages_i.append({'role': roles[j%2], 'content': message})
                messages.append(messages_i)
        else:
            for query in args.query:
                if args.system == None:
                    messages.append([{"role": "user", "content": query}])
                else:
                    messages.append([{"role": "system", "content": args.system},
                                    {"role": "user", "content": query}])
    else:
        if args.messages != None:
            roles = ['user', 'model']
            for i in range(len(args.messages)):
                messages_i = []
                assert len(args.messages[i]) % 2 == 1
                for j, message in enumerate(args.messages[i]):
                    messages_i.append({'role': roles[j%2], 'parts': [message]})
                messages.append(messages_i)
        else:
            for query in args.query:
                messages.append([{"role": "user", "parts": query}])
    return messages


class Gemini:
    def __init__(self, model="gemini-1.5-flash", N=3):
        self.model = genai.GenerativeModel(model)
        self.count_limit = N
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def get_response(self, messages, system_instruction = None, n = 1) -> str:
        # Query the model
        records = []
        counts = 0
        while len(records) < n and counts < self.count_limit:
            if system_instruction != None:
                self.system_instruction = system_instruction
            try:
                num = n
                records = []
                while (len(records) < n):
                    time_begin = time.time()
                    response = self.model.generate_content(messages, 
                                                        safety_settings=self.safety_settings,
                                                        generation_config=genai.types.GenerationConfig(candidate_count = num)) 
                    time_completion = time.time()
                    for i in range(0, num):
                        if response.candidates[i].finish_reason == 1:
                            output = response.candidates[i].content.parts[0].text.strip()
                            completion_tokens = self.model.count_tokens(output).total_tokens
                            usage = {"prompt_tokens": response.usage_metadata.prompt_token_count,
                                    "completion_tokens": completion_tokens,
                                    "time_prompt": 0,
                                    "time_completion": time_completion - time_begin}
                            record = {"output": output, "usage": usage}
                            records.append(record)
                    num = n - len(record)
                    
            except Exception as error:
                response = ""
                print(error)
                print('Sleeping for 10 seconds')
                time.sleep(10)
            counts += 1
        if counts == self.count_limit:
            raise EOFError("<Exceed the count limit>")
        return records


def load_model(args):
    if args.model_type == "vllm":
        from vllm import LLM
        args.model = LLM(model=args.model_name, trust_remote_code=True, tensor_parallel_size=len(args.gpu.split(",")), dtype = args.dtype)
    elif args.model_type == "gemini":
        genai.configure(api_key=args.google_api_key)
        args.model = Gemini(model=args.model_name)
    elif args.model_type == "openai":
        args.client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)


def gpt_parallel_generate(args, message):
    records = []
    res = args.client.chat.completions.create(
        model=args.model_name,
        messages = message,
        n = args.num,
        logprobs = True,
        max_tokens = args.max_new_tokens
        )
    for i in range(0, args.num):
        output = res.choices[i].message.content
        completion_tokens = len(res.choices[i].logprobs.content)
        usage = {
                    "prompt_tokens": res.usage.prompt_tokens,
                    "completion_tokens": completion_tokens
                }
        record = {
                    "output": output, 
                    "usage": usage
                }
        records.append(record)
    return records


def LLM_generate(args):
    """
    Generate responses using different LLM backends with parallel processing support.
    
    Args:
        args: Arguments containing model configuration and generation parameters
        
    Returns:
        List of generated records with outputs and usage statistics
    """
    
    assert args.messages != None or args.query != None
    messages = []
    messages = get_messages(args)
    outputs = []
    records = []
    prompt_tokens = []
    completion_tokens = []
    if args.model_type == "gemini":
        if args.max_num_workers == 1:
            for i in tqdm(range(0, len(messages))):
                records_ = args.model.get_response(messages[i], args.system, args.num)
                records.append(records_)
        else:
            records = [None] * len(messages) 
            with ThreadPoolExecutor(max_workers=args.max_num_workers) as executor:
                future_to_index = {}
                for index, message in enumerate(messages):
                    future = executor.submit(args.model.get_response, message, args.system, args.num)
                    future_to_index[future] = index
                with tqdm(total=len(messages)) as pbar:
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]  
                        try:
                            data = future.result()
                            records[idx] = data  
                            pbar.update(1)
                        except Exception as exc:
                            print(f'Index {idx} generated exception: {exc}')
                            pdb.set_trace()    
                
        for i in range(0, len(messages)):
            for j in range(0, args.num):
                if "tot" in args.reasoning:
                    output_key = parse_best_solution(records[i][j]["output"])
                else:
                    output_key = parse_answer(args, records[i][j]["output"])
                records[i][j]["output_key"] = output_key
                
    elif args.model_type == "openai":
        if args.max_num_workers == 1:
            for i in tqdm(range(0, len(messages))):
                records_i = gpt_parallel_generate(args, messages[i])
                records.append(records_i)
        else:
            records = [None] * len(messages) 
            with ThreadPoolExecutor(max_workers=args.max_num_workers) as executor:
                future_to_index = {}
                for index, message in enumerate(messages):
                    future = executor.submit(gpt_parallel_generate, args, message)
                    future_to_index[future] = index
                with tqdm(total=len(messages)) as pbar:
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]  
                        try:
                            data = future.result()
                            records[idx] = data  
                            pbar.update(1)
                        except Exception as exc:
                            print(f'Index {idx} generated exception: {exc}')
                            pdb.set_trace()    
                        
        for i in range(0, len(messages)):
            for j in range(0, args.num):
                if "tot" in args.reasoning:
                    output_key = parse_best_solution(records[i][j]["output"])
                else:
                    output_key = parse_answer(args, records[i][j]["output"])
                records[i][j]["output_key"] = output_key
                    
    elif args.model_type == "vllm":
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=args.temperature, n = args.num, max_tokens = args.max_new_tokens, top_p = 0.9)
        res = args.model.chat(messages=messages, sampling_params=sampling_params)
        for i in range(0, len(res)):
            outputs.append([output.text for output in res[i].outputs])
            prompt_tokens.append(len(res[i].prompt_token_ids))
            completion_tokens.append([len(output.token_ids) for output in res[i].outputs])
        
        for i in range(0, len(outputs)):
            outputs_i = outputs[i]
            completion_tokens_i = completion_tokens[i]
            prompt_tokens_i = prompt_tokens[i]
            records_i = []
            for j in range(0, args.num):
                usage = {"prompt_tokens": prompt_tokens_i,
                        "completion_tokens": completion_tokens_i[j]
                        }
                output = outputs_i[j]
                if "tot" in args.reasoning:
                    output_key = parse_best_solution(output)
                else:
                    output_key = parse_answer(args, output)
                record = {"output": output, "output_key": output_key, "usage": usage}
                records_i.append(record)
            records.append(records_i)
    else:
        raise NotImplementedError(f"Model type \"{args.model_type}\" not supported (should be \"vllm\", \"gemini\" or \"openai\").")
    return records 
