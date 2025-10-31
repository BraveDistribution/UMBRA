#!/bin/bash
# Extract embeddings from all 4 pretrained checkpoints for finetuning dataset (ISLES)
#
# IMPORTANT: This extracts from the FULL finetuning dataset using PRETRAINED checkpoints.
# This is ONLY safe because ISLES was NOT included in the pretraining data.

# Configuration
DATA_DIR="/home/mg873uh/data/Petros/isles"  # UPDATE THIS PATH to your ISLES dataset location
OUTPUT_BASE="/home/mg873uh/data/Petros/isles_embeddings"
SEED=42
MODALITIES="dwi flair adc"  # ISLES modalities
SPLIT="full"  # Extract from full dataset (safe with pretrained checkpoints)

# Checkpoint paths (pretrained, NOT finetuned)
CHECKPOINT_MCL="/home/mg873uh/data/Petros/ckpts/contrastive_modality-step=200000.ckpt"
CHECKPOINT_CL="/home/mg873uh/data/Petros/ckpts/contrastive_regular-step=200000.ckpt"
CHECKPOINT_MAE_MCL="/home/mg873uh/data/Petros/ckpts/combined_modality-step=200000.ckpt"
CHECKPOINT_MAE_CL="/home/mg873uh/data/Petros/ckpts/combined_regular-step=200000.ckpt"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script to point to your ISLES dataset."
    exit 1
fi

echo "=========================================="
echo "EMBEDDING EXTRACTION - FINETUNING DATASET"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Output base: $OUTPUT_BASE"
echo "Modalities: $MODALITIES"
echo "Split: $SPLIT"
echo "Seed: $SEED"
echo "=========================================="
echo ""

# Extract embeddings for each checkpoint
echo "Extracting embeddings for MCL (Modality Contrastive Learning)..."
python extract_embeddings_finetune.py \
    --checkpoint "$CHECKPOINT_MCL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/mcl" \
    --modalities $MODALITIES \
    --split $SPLIT \
    --seed $SEED

echo ""
echo "Extracting embeddings for CL (Contrastive Learning)..."
python extract_embeddings_finetune.py \
    --checkpoint "$CHECKPOINT_CL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/cl" \
    --modalities $MODALITIES \
    --split $SPLIT \
    --seed $SEED

echo ""
echo "Extracting embeddings for MAE + MCL (Combined)..."
python extract_embeddings_finetune.py \
    --checkpoint "$CHECKPOINT_MAE_MCL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/mae_mcl" \
    --modalities $MODALITIES \
    --split $SPLIT \
    --seed $SEED

echo ""
echo "Extracting embeddings for MAE + CL (Combined)..."
python extract_embeddings_finetune.py \
    --checkpoint "$CHECKPOINT_MAE_CL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/mae_cl" \
    --modalities $MODALITIES \
    --split $SPLIT \
    --seed $SEED

echo ""
echo "=========================================="
echo "All extractions complete!"
echo "=========================================="
echo "Results saved in: $OUTPUT_BASE"
echo ""
echo "Output structure:"
echo "  $OUTPUT_BASE/mcl/embeddings.pt"
echo "  $OUTPUT_BASE/mcl/metadata.csv"
echo "  $OUTPUT_BASE/cl/embeddings.pt"
echo "  $OUTPUT_BASE/cl/metadata.csv"
echo "  $OUTPUT_BASE/mae_mcl/embeddings.pt"
echo "  $OUTPUT_BASE/mae_mcl/metadata.csv"
echo "  $OUTPUT_BASE/mae_cl/embeddings.pt"
echo "  $OUTPUT_BASE/mae_cl/metadata.csv"
echo "=========================================="
