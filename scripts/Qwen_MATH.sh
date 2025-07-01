python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning DiP \
    --shot 0 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 32  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning CoT \
    --shot 0 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 32  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning L2M \
    --shot 1 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 32  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning SBP \
    --shot 0 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 32  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning AnP \
    --shot 1 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 32  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning ToT \
    --shot 3 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 5  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning ToT \
    --shot 5 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 5  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning ToT \
    --shot 10 \
    --batchsize 10 \
    --range_begin 0 \
    --range_end 5  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning S-RF \
    --shot 0 \
    --batchsize 5 \
    --range_begin 0 \
    --range_end 1  \
    --gpu 0,1

python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset MATH \
    --reasoning MAD \
    --shot 0 \
    --batchsize 1 \
    --range_begin 0 \
    --range_end 1  \
    --gpu 0,1
