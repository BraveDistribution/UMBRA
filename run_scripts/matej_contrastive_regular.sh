#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=umbra
#SBATCH --ntasks=1
#SBATCH --output=umbra-test.log

source /home/mg873uh/Projects/UMBRA/.venv/bin/activate
cd /home/mg873uh/Projects/UMBRA


wandb login

python pretrain.py \
  --data_dir="/home/mg873uh/Projects_kb/data/pretrain_parsed" \
  --model_checkpoint_dir="/home/mg873uh/Projects_mg/checkpoints" \
  --pretraining_mode=contrastive_only \
  --contrastive_mode=regular \
  --experiment_name=contrastive_regular \
  --input_size=96 \
  --batch_size=14 \
  --steps=250000 \
  --accumulate_grad_batches=4 \
  --learning_rate=1e-4 \
  --num_checkpoints=10 \
  --fast_dev_run=False \

