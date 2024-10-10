#!/bin/bash

# python main.py \
#     --model_name "${1}" \
#     --inference_base_model \
#     --wandb_token "${2}"
python main.py --model_name unsloth/llama-3-8b-bnb-4bit --inference_base_model  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 > submission/llama-3-8b-bnb-4bit_inference_result.json
python main.py --model_name unsloth/mistral-7b-v0.3-bnb-4bit --inference_base_model  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 > submission/mistral-7b-v0.3-bnb-4bit_inference_result.json
python main.py --model_name unsloth/tinyllama-bnb-4bit --inference_base_model  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 > submission/tinyllama-bnb-4bit_inference_result.json
python main.py --model_name unsloth/gemma-2b-bnb-4bit --inference_base_model  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6 > submission/gemma-2b-bnb-4bit_inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/DPO_20240606-181550/checkpoint-795 > outputs/DPO_20240606-181550/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/DPO_20240607-011953/checkpoint-795 > outputs/DPO_20240607-011953/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/DPO_20240607-093653/checkpoint-2385 > outputs/DPO_20240607-093653/checkpoint-2385/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/DPO_20240607-150533/checkpoint-3975 > outputs/DPO_20240607-150533/checkpoint-3975/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/DPO_20240608-000822/checkpoint-795 > outputs/DPO_20240608-000822/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/DPO_20240608-084000/checkpoint-795 > outputs/DPO_20240608-084000/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/ORPO_20240606-181127/checkpoint-795 > outputs/ORPO_20240606-181127/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/ORPO_20240607-054943/checkpoint-795 > outputs/ORPO_20240607-054943/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/ORPO_20240607-123511/checkpoint-2385 > outputs/ORPO_20240607-123511/checkpoint-2385/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/ORPO_20240607-195934/checkpoint-3975 > outputs/ORPO_20240607-195934/checkpoint-3975/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/ORPO_20240608-044516/checkpoint-795 > outputs/ORPO_20240608-044516/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model outputs/ORPO_20240608-104834/checkpoint-795 > outputs/ORPO_20240608-104834/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model output_DPO_tiny_beta0.2/DPO_20240610-182905/checkpoint-795 > output_DPO_tiny_beta0.2/DPO_20240610-182905/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model output_DPO_tiny_beta0.05/DPO_20240608-235356/checkpoint-795 > output_DPO_tiny_beta0.05/DPO_20240608-235356/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model output_DPO_tiny_lrlinear/DPO_20240608-235730/checkpoint-795 > output_DPO_tiny_lrlinear/DPO_20240608-235730/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model output_ORPO_tiny_beta0.2/ORPO_20240609-014542/checkpoint-795 > output_ORPO_tiny_beta0.2/ORPO_20240609-014542/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model output_ORPO_tiny_beta0.05/ORPO_20240610-182754/checkpoint-795 > output_ORPO_tiny_beta0.05/ORPO_20240610-182754/checkpoint-795/inference_result.json
python main.py  --wandb_token ada44c2392487a6bc1cf2894e743a16024acb2a6  --use_finetuned --finetuned_model output_ORPO_tiny_lrlinear/ORPO_20240609-014938/checkpoint-795 > output_ORPO_tiny_lrlinear/ORPO_20240609-014938/checkpoint-795/inference_result.json