prompt_format = " Your final result should be in the form \\boxed{answer}, at the end of your response."

directly_answer = "{question} You should only answer the final result with a single numerical number. Do not say other words."

io = "{question}" + prompt_format

io_briefly = io + " You should answer with no more than 200 words."

cot_pre = "Please answer the given question." + prompt_format + '\n\n'

cot_0_shot = cot_pre + '''Question: {question} 
Answer: Let's think step by step.'''

# cot_1_shot = cot_pre + '''Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
# Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is \\boxed{72}.

# Question: {question}
# Answer:
# '''

Least_to_Most_0_shot = cot_pre + ''' In order to solve the question more conveniently and efficiently, break down the question into progressive sub-questions. Answer the sub-questions and get the final result according to sub-questions and their answers.

Question: {question}
Answer:
''' 

tot_post = '''
Given the question and several solutions, decide which solution is the most promising. Analyze each solution in detail, then conclude in the last line "The index of the best solution is x", where x is the index number of the solution.'''

tot_3_solutions = '''
Question: {question}

Solution 1: {solution1}
Solution 2: {solution2}
Solution 3: {solution3}'''

tot_5_solutions = tot_3_solutions + '''
Solution 4: {solution4}
Solution 5: {solution5}'''

tot_10_solutions = tot_5_solutions + '''
Solution 6: {solution6}
Solution 7: {solution7}
Solution 8: {solution8}
Solution 9: {solution9}
Solution 10: {solution10}'''

anologous_1_prompt = '''Your task is to tackle mathematical problems. When presented with a math problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.

# Initial Problem:
{question}

# Instructions:
## Relevant Problems:
Recall an example of the math problem that is relevant to the initial problem. Your problem should be distinct from the initial problem (e.g., involving different numbers and names). For the example problem:
- After "Q: ", describe the problem.
- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{}.

## Solve the Initial Problem:
Q: Copy and paste the initial problem here.
A: Explain the solution and enclose the ultimate answer in \\boxed{} here.
'''

anologous_3_prompt = '''Your task is to tackle mathematical problems. When presented with a math problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.

# Initial Problem:
{question}

# Instructions:
## Relevant Problems:
Recall three examples of math problems that are relevant to the initial problem. Your problems should be distinct from each other and from the initial problem (e.g., involving different numbers and names). For each problem:
- After "Q: ", describe the problem.
- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{}.

## Solve the Initial Problem:
Q: Copy and paste the initial problem here.
A: Explain the solution and enclose the ultimate answer in \\boxed{} here.
'''

anologous_5_prompt = '''Your task is to tackle mathematical problems. When presented with a math problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.

# Initial Problem:
{question}

# Instructions:
## Relevant Problems:
Recall five examples of math problems that are relevant to the initial problem. Your problems should be distinct from each other and from the initial problem (e.g., involving different numbers and names). For each problem:
- After "Q: ", describe the problem.
- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{}.

## Solve the Initial Problem:
Q: Copy and paste the initial problem here.
A: Explain the solution and enclose the ultimate answer in \\boxed{} here.
'''

SBP_extract = '''You are an expert at mathematics. Your task is to extract the mathematics concepts and principles involved in solving the problem.
Question:
{question}

Principles involved:
'''

SBP_answer = "You are an expert at mathematics. You are given a mathematics problem and a set of principles involved in solving the problem. Solve the problem step by step by following the principles." + prompt_format + '''
Question:
{question}
 
Principles:
{principles}

Answer:
'''

