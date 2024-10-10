# AI Homework 6

## Install Necessary Packages
cuda version : 12.1
```bash=
conda create -y -n hw6 python=3.10
conda activate hw6
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --no-deps trl peft accelerate bitsandbytes xformers==0.0.23
pip install tqdm packaging wandb
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Training 
```bash=
cd AI2024-hw6
sh run.sh
```

## Evaluation 
Inference the base models and the corresponding fine-tuned models.
```bash=
cd AI2024-hw6
sh inference.sh
```

## Setting and hyperparameters used in each experiments
The corresponding generated texts have been saved in `submissions` folder with the label listed below.
| Label | strategy | base model |epoch|lr_schduler_type|beta|
| -------- | -------- | -------- |-------- |-------- |-------- |
| DPO_llama-3-8b-bnb-4bit_20240607-011953| DPO|unsloth/llama-3-8b-bnb-4bit|1|cosine|0.1|
| ORPO_llama-3-8b-bnb-4bit_20240607-054943|ORPO|unsloth/llama-3-8b-bnb-4bit|1|cosine|0.1|
| DPO_mistral-7b-v0.3-bnb-4bit_20240608-000822|DPO|unsloth/mistral-7b-v0.3-bnb-4bit|1|cosine|0.1|
| ORPO_mistral-7b-v0.3-bnb-4bit_20240608-044516|ORPO|unsloth/mistral-7b-v0.3-bnb-4bit|1|cosine|0.1|
| DPO_gemma-2b-bnb-4bit_20240608-084000|DPO|unsloth/gemma-2b-bnb-4bit|1|cosine|0.1|
| ORPO_gemma-2b-bnb-4bit_20240608-104834|ORPO|unsloth/gemma-2b-bnb-4bit|1|cosine|0.1|
| DPO_tinyllama-bnb-4bit_20240606-181550|DPO|unsloth/tinyllama-bnb-4bit|1|cosine|0.1|
| ORPO_tinyllama-bnb-4bit_20240606-181127|ORPO|unsloth/tinyllama-bnb-4bit|1|cosine|0.1|
| DPO_tinyllama-bnb-4bit_20240607-093653|DPO|unsloth/tinyllama-bnb-4bit|3|cosine|0.1|
| ORPO_tinyllama-bnb-4bit_20240607-123511|ORPO|unsloth/tinyllama-bnb-4bit|3|cosine|0.1|
| DPO_tinyllama-bnb-4bit_20240607-150533|DPO|unsloth/tinyllama-bnb-4bit|5|cosine|0.1|
| ORPO_tinyllama-bnb-4bit_20240607-195934|ORPO|unsloth/tinyllama-bnb-4bit|5|cosine|0.1|
| DPO_tinyllama-bnb-4bit_20240608-235730|DPO|unsloth/tinyllama-bnb-4bit|1|linear|0.1|
| ORPO_tinyllama-bnb-4bit_20240609-014938|ORPO|unsloth/tinyllama-bnb-4bit|1|linear|0.1|
| DPO_tinyllama-bnb-4bit_20240610-182905|DPO|unsloth/tinyllama-bnb-4bit|1|cosine|0.2|
| ORPO_tinyllama-bnb-4bit_20240609-014542|ORPO|unsloth/tinyllama-bnb-4bit|1|cosine|0.2|
| DPO_tinyllama-bnb-4bit_20240608-235356|DPO|unsloth/tinyllama-bnb-4bit|1|cosine|0.05|
| ORPO_tinyllama-bnb-4bit_20240610-182754|ORPO|unsloth/tinyllama-bnb-4bit|1|cosine|0.05|
