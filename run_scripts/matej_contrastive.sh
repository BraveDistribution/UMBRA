#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=umbra
#SBATCH --output=umbra-test.log

source /home/mg873uh/Projects/UMBRA/.venv/bin/activate
cd /home/mg873uh/Projects/UMBRA

wandb login

python pretrain.py \
  --data_dir="/home/mg873uh/Projects_kb/data/pretrain_parsed" \
  --model_checkpoint_dir="/home/mg873uh/Projects_mg/checkpoints" \
  --pretraining_mode=contrastive_only \
  --contrastive_mode=modality_pairs \
  --experiment_name=contrastive_modality \
  --input_size=96 \
  --batch_size=14 \
  --num_workers=14 \
  --epochs=100 \
  --accumulate_grad_batches=2 \
  --learning_rate=1e-4 \
  --num_checkpoints=10 \
  --fast_dev_run=False \
  --devices=2

