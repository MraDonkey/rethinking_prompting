python main.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --split test \
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
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
    --dataset GSM8K \
    --reasoning MAD \
    --shot 0 \
    --batchsize 1 \
    --range_begin 0 \
    --range_end 1  \
    --gpu 0,1
