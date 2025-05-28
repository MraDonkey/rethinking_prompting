prompt_format = " Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response."

directly_answer = "{question} You should only answer the final result with a single numerical number. Do not say other words."

io = "{question}" + prompt_format

io_briefly = io + " You should answer with no more than 200 words."

cot_pre = "Please answer the given question." + prompt_format + '\n\n'

cot_0_shot = cot_pre + '''Question: {question} Let's think step by step.
Answer:'''

# cot_1_shot = cot_pre + '''Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
# Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is \\boxed{72}.

# Question: {question}
# Answer:
# '''

cot_1_shot = cot_pre + '''Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is \\boxed{72}.

Question: {question}
Answer:
'''

cot_5_shot = cot_pre + '''Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is \\boxed{72}.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. The answer is \\boxed{10}.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. The answer is \\boxed{5}.

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Maila read 12 x 2 = <<12*2=24>>24 pages today. So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. There are 120 - 36 = <<120-36=84>>84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages. The answer is \\boxed{42}.

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 3*2=<<3*2=6>>6 pages a week. So he writes 6*2=<<6*2=12>>12 pages every week. That means he writes 12*52=<<12*52=624>>624 pages a year. The answer is \\boxed{624}.

Question: {question}
Answer:
'''

Least_to_Most_1_shot = cot_pre + '''Question: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?
Answer: Let's break down this problem: 1. How many apples does Anna have? 2. How many apples do they have together?
1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.
2. Elsa and Anna have 5 + 7 = 12 apples together.
The answer is: \\boxed{12}.

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

