prompt_format = "  The given information may not conform to common sense and the result may be a nonsense decimal or negative number, it's okay, output it instead of considering it is unreasonable. Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response."

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

cot_1_shot = cot_pre + '''Question: Natalia sold clips to 48564 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48564/2 = <<48564/2=24282>>24282 clips in May. Natalia sold 48564+24282 = <<48564+24282=72846>>72846 clips altogether in April and May. The answer is \\boxed{72846}.

Question: {question}
Answer:
'''

cot_5_shot = cot_pre + '''Question: Natalia sold clips to 48564 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48564/2 = <<48564/2=24282>>24282 clips in May. Natalia sold 48564+24282 = <<48564+24282=72846>>72846 clips altogether in April and May. The answer is \\boxed{72846}.

Question: Weng earns $1293 an hour for babysitting. Yesterday, she just did 612 minutes of babysitting. How much did she earn?
Answer: Weng earns 1293/60 = $<<1293/60=21.55>>21.55 per minute. Working 612 minutes, she earned 21.55 x 612 = $<<21.55*612=13188.6>>13188.6. The answer is \\boxed{13188.6}.

Question: Betty is saving money for a new wallet which costs $8200. Betty has only half of the money she needs. Her parents decided to give her $1525 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
Answer: In the beginning, Betty has only 8200 / 2 = $<<8200/2=4100>>4100. Betty's grandparents gave her 1525 * 2 = $<<1525*2=3050>>3050. This means, Betty needs 8200 - 4100 - 3050 - 1525 = $<<8200-4100-3050-1525=-475>>-475 more. The answer is \\boxed{-475}.

Question: Julie is reading a 12602-page book. Yesterday, she was able to read 3127 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Maila read 3127 x 2 = <<3127*2=6254>>6254 pages today. So she was able to read a total of 3127 + 6254 = <<3127+6254=9381>>9381 pages since yesterday. There are 12602 - 9381 = <<12602-9381=3221>>3221 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 3221/2 = <<3221/2=1610.5>>1610.5 pages. The answer is \\boxed{1610.5}.

Question: James writes a 312996-page letter to 2143 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 312996*2143=<<312996*2143=670750428>>670750428 pages a week. So he writes 670750428*2=<<670750428*2=1341500856>>1341500856 pages every week. That means he writes 1341500856*52=<<1341500856*52=69758044512>>69758044512 pages a year. The answer is \\boxed{69758044512}.

Question: {question}
Answer:
'''

Least_to_Most_1_shot = cot_pre + '''Question: Elsa has 524866 apples. Anna has 432343 more apples than Elsa. How many apples do they have together?
Answer: Let's break down this problem: 1. How many apples does Anna have? 2. How many apples do they have together?
1. Anna has 432343 more apples than Elsa. So Anna has 524866 + 432343 = 957209 apples.
2. Elsa and Anna have 524866 + 957209 = 1482075 apples together.
The answer is: \\boxed{1482075}.

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

