#!/bin/bash

# python main.py \
#     --exp_name "${1}" \
#     --model_name "${2}" \
#     --train \
#     --wandb_token "${3}" \
#     --num_epochs 1 \

python main.py --exp_name DPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name ORPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name DPO --model_name unsloth/llama-3-8b-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name ORPO --model_name unsloth/llama-3-8b-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name DPO --model_name unsloth/mistral-7b-v0.3-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name ORPO --model_name unsloth/mistral-7b-v0.3-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name DPO --model_name unsloth/gemma-2b-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 
python main.py --exp_name ORPO --model_name unsloth/gemma-2b-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 

python main.py --exp_name DPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 3
python main.py --exp_name ORPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 3 
python main.py --exp_name DPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 5 
python main.py --exp_name ORPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 5 

python main.py --exp_name DPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 --beta 0.05 --output_dir ./output_DPO_tiny_beta0.05
python main.py --exp_name ORPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 --beta 0.05 --output_dir ./output_DPO_tiny_beta0.05
python main.py --exp_name DPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 --beta 0.2 --output_dir ./output_ORPO_tiny_beta0.2
python main.py --exp_name ORPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 --beta 0.2 --output_dir ./output_ORPO_tiny_beta0.2

python main.py --exp_name DPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 --lr_scheduler_type linear --output_dir ./output_DPO_tiny_lrlinear
python main.py --exp_name ORPO --model_name unsloth/tinyllama-bnb-4bit  --train --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 --num_epochs 1 --lr_scheduler_type linear --output_dir ./output_ORPO_tiny_lrlinear