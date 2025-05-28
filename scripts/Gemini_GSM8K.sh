## You can control the maximum concurrency level by adjusting the "max_num_workers" parameter.

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning DiP \
    --shot 0 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 8  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning CoT \
    --shot 0 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 8  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning L2M \
    --shot 1 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 8  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning SBP \
    --shot 0 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 8  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning AnP \
    --shot 1 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 8  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

####  

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning DiP \
    --shot 0 \
    --batch_size 10 \
    --range_begin 8 \
    --range_end 16  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning CoT \
    --shot 0 \
    --batch_size 10 \
    --range_begin 8 \
    --range_end 16  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning L2M \
    --shot 1 \
    --batch_size 10 \
    --range_begin 8 \
    --range_end 16  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning SBP \
    --shot 0 \
    --batch_size 10 \
    --range_begin 8 \
    --range_end 16  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning AnP \
    --shot 1 \
    --batch_size 10 \
    --range_begin 8 \
    --range_end 16  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key



python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning ToT \
    --shot 3 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 5  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning ToT \
    --shot 5 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 5  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning ToT \
    --shot 10 \
    --batch_size 10 \
    --range_begin 0 \
    --range_end 5  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning S-RF \
    --shot 0 \
    --batch_size 5 \
    --range_begin 0 \
    --range_end 1  \
    --max_num_workers 10 \
    --gemini_api_key your_api_key

python main.py \
    --model_name gemini-1.5-flash \
    --model_type gemini \
    --split test \
    --dataset GSM8K \
    --reasoning MAD \
    --shot 0 \
    --batch_size 1 \
    --range_begin 0 \
    --range_end 1  \
    --max_num_workers 1 \
    --gemini_api_key your_api_key