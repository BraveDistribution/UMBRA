#!/bin/bash
# Extract embeddings from all 4 checkpoints

# Configuration
DATA_DIR="/home/mg873uh/Projects_kb/data/pretrain_parsed"
OUTPUT_BASE="/home/mg873uh/Projects/UMBRA/embeddings_analysis"
SEED=42

# Checkpoint paths
CHECKPOINT_MCL="/home/mg873uh/data/Petros/ckpts/contrastive_modality-step=200000.ckpt"
CHECKPOINT_CL="/home/mg873uh/data/Petros/ckpts/contrastive_regular-step=200000.ckpt"
CHECKPOINT_MAE_MCL="/home/mg873uh/data/Petros/ckpts/combined_modality-step=200000.ckpt"
CHECKPOINT_MAE_CL="/home/mg873uh/data/Petros/ckpts/combined_regular-step=200000.ckpt"

# Extract embeddings for each checkpoint
echo "Extracting embeddings for MCL..."
python extract_embeddings.py \
    --checkpoint "$CHECKPOINT_MCL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/mcl" \
    --seed $SEED

echo "Extracting embeddings for CL..."
python extract_embeddings.py \
    --checkpoint "$CHECKPOINT_CL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/cl" \
    --seed $SEED

echo "Extracting embeddings for MAE + MCL..."
python extract_embeddings.py \
    --checkpoint "$CHECKPOINT_MAE_MCL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/mae_mcl" \
    --seed $SEED

echo "Extracting embeddings for MAE + CL..."
python extract_embeddings.py \
    --checkpoint "$CHECKPOINT_MAE_CL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/mae_cl" \
    --seed $SEED

echo ""
echo "All extractions complete!"
echo "Results saved in: $OUTPUT_BASE"
