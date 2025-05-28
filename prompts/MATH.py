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

cot_1_shot = cot_pre + '''Question: Ryan has 3 red lava lamps and 3 blue lava lamps. He arranges them in a row on a shelf randomly, then turns 3 random lamps on. What is the probability that the leftmost lamp on the shelf is red, and the leftmost lamp which is turned on is also red?  
Answer: There are $\\binom{6}{3}=20$ ways for Ryan to arrange the lamps, and $\\binom{6}{3}=20$ ways for him to choose which lamps are on, giving $20\\cdot20=400$ total possible outcomes. There are two cases for the desired outcomes: either the left lamp is on, or it isn't. If the left lamp is on, there are $\\binom{5}{2}=10$ ways to choose which other lamps are on, and $\\binom{5}{2}=10$ ways to choose which other lamps are red. This gives $10\\cdot10=100$ possibilities. If the first lamp isn't on, there are $\\binom{5}{3}=10$ ways to choose which lamps are on, and since both the leftmost lamp and the leftmost lit lamp must be red, there are $\\binom{4}{1}=4$ ways to choose which other lamp is red. This case gives 40 valid possibilities, for a total of 140 valid arrangements out of 400. Therefore, the probability is $\\dfrac{140}{400}=\\boxed{\\dfrac{7}{20}}$.

Question: {question}
Answer:
'''

cot_5_shot = cot_pre + '''Question: Ryan has 3 red lava lamps and 3 blue lava lamps. He arranges them in a row on a shelf randomly, then turns 3 random lamps on. What is the probability that the leftmost lamp on the shelf is red, and the leftmost lamp which is turned on is also red?  
Answer: There are $\\binom{6}{3}=20$ ways for Ryan to arrange the lamps, and $\\binom{6}{3}=20$ ways for him to choose which lamps are on, giving $20\\cdot20=400$ total possible outcomes. There are two cases for the desired outcomes: either the left lamp is on, or it isn't. If the left lamp is on, there are $\\binom{5}{2}=10$ ways to choose which other lamps are on, and $\\binom{5}{2}=10$ ways to choose which other lamps are red. This gives $10\\cdot10=100$ possibilities. If the first lamp isn't on, there are $\\binom{5}{3}=10$ ways to choose which lamps are on, and since both the leftmost lamp and the leftmost lit lamp must be red, there are $\\binom{4}{1}=4$ ways to choose which other lamp is red. This case gives 40 valid possibilities, for a total of 140 valid arrangements out of 400. Therefore, the probability is $\\dfrac{140}{400}=\\boxed{\\dfrac{7}{20}}$.

Question: On the $xy$-plane, the origin is labeled with an $M$. The points $(1,0)$, $(-1,0)$, $(0,1)$, and $(0,-1)$ are labeled with $A$'s. The points $(2,0)$, $(1,1)$, $(0,2)$, $(-1, 1)$, $(-2, 0)$, $(-1, -1)$, $(0, -2)$, and $(1, -1)$ are labeled with $T$'s. The points $(3,0)$, $(2,1)$, $(1,2)$, $(0, 3)$, $(-1, 2)$, $(-2, 1)$, $(-3, 0)$, $(-2,-1)$, $(-1,-2)$, $(0, -3)$, $(1, -2)$, and $(2, -1)$ are labeled with $H$'s. If you are only allowed to move up, down, left, and right, starting from the origin, how many distinct paths can be followed to spell the word MATH?
Answer: From the M, we can proceed to four different As. Note that the letters are all symmetric, so we can simply count one case (say, that of moving from M to the bottom A) and then multiply by four.\n\nFrom the bottom A, we can proceed to any one of three Ts. From the two Ts to the sides of the A, we can proceed to one of two Hs. From the T that is below the A, we can proceed to one of three Hs. Thus, this case yields $2 \\cdot 2 + 3 = 7$ paths.\n\nThus, there are $4 \\cdot 7 = \\boxed{28}$ distinct paths.

Question: Factor the following expression: $55z^{17}+121z^{34}$.
Answer: The greatest common factor of the two coefficients is $11$, and the greatest power of $z$ that divides both terms is $z^{17}$. So, we factor $11z^{17}$ out of both terms:\n\n\\begin{align*}\n55z^{17}+121z^{34} &= 11z^{17}\\cdot 5 +11z^{17}\\cdot 11z^{17}\\\\\n&= \\boxed{11z^{17}(5+11z^{17})}\n\\end{align*}.

Question: Allen and Ben are painting a fence. The ratio of the amount of work Allen does to the amount of work Ben does is $3:5$. If the fence requires a total of $240$ square feet to be painted, how many square feet does Ben paint?
Answer: Between them, Allen and Ben are dividing the work into $8$ equal parts, $3$ of which Allen does and $5$ of which Ben does. Each part of the work requires $\\frac{240}{8} = 30$ square feet to be painted. Since Ben does $5$ parts of the work, he will paint $30 \\cdot 5 = \\boxed{150}$ square feet of the fence.

Question: Suppose $z$ and $w$ are complex numbers such that\n\\[|z| = |w| = z \\overline{w} + \\overline{z} w= 1.\\]Find the largest possible value of the real part of $z + w.$
Answer: Let $z = a + bi$ and $w = c + di,$ where $a,$ $b,$ $c,$ and $d$ are complex numbers.  Then from $|z| = 1,$ $a^2 + b^2 = 1,$ and from $|w| = 1,$ $c^2 + d^2 = 1.$  Also, from $z \\overline{w} + \\overline{z} w = 1,$\n\\[(a + bi)(c - di) + (a - bi)(c + di) = 1,\\]so $2ac + 2bd = 1.$\n\nThen\n\\begin{align*}\n(a + c)^2 + (b + d)^2 &= a^2 + 2ac + c^2 + b^2 + 2bd + d^2 \\\\\n&= (a^2 + b^2) + (c^2 + d^2) + (2ac + 2bd) \\\\\n&= 3.\n\\end{align*}The real part of $z + w$ is $a + c,$ which can be at most $\\sqrt{3}.$  Equality occurs when $z = \\frac{\\sqrt{3}}{2} + \\frac{1}{2} i$ and $w = \\frac{\\sqrt{3}}{2} - \\frac{1}{2} i,$ so the largest possible value of $a + c$ is $\\boxed{\\sqrt{3}}.$

Question: {question}
Answer:
'''

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

