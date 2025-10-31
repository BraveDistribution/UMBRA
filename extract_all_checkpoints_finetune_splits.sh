#!/bin/bash
# Extract embeddings from all 4 pretrained checkpoints for ISLES dataset
# Processes train/val/test splits separately
#
# IMPORTANT: This extracts using PRETRAINED checkpoints (not finetuned).
# This is safe because ISLES was NOT included in the pretraining data.

# Configuration
ISLES_BASE="/home/mg873uh/data/Petros/isles"  # UPDATE THIS: Base ISLES directory containing train/val/test
OUTPUT_BASE="/home/mg873uh/data/Petros/isles_embeddings"
SEED=42
MODALITIES="dwi flair adc"  # ISLES modalities

# Checkpoint paths (pretrained, NOT finetuned)
CHECKPOINT_MCL="/home/mg873uh/data/Petros/ckpts/contrastive_modality-step=200000.ckpt"
CHECKPOINT_CL="/home/mg873uh/data/Petros/ckpts/contrastive_regular-step=200000.ckpt"
CHECKPOINT_MAE_MCL="/home/mg873uh/data/Petros/ckpts/combined_modality-step=200000.ckpt"
CHECKPOINT_MAE_CL="/home/mg873uh/data/Petros/ckpts/combined_regular-step=200000.ckpt"

# Verify base directory exists
if [ ! -d "$ISLES_BASE" ]; then
    echo "ERROR: ISLES base directory not found: $ISLES_BASE"
    echo "Please update ISLES_BASE in this script."
    exit 1
fi

# Define splits to process
SPLITS=("train" "val" "test")

echo "=========================================="
echo "EMBEDDING EXTRACTION - ISLES DATASET"
echo "=========================================="
echo "Base directory: $ISLES_BASE"
echo "Output base: $OUTPUT_BASE"
echo "Modalities: $MODALITIES"
echo "Splits: ${SPLITS[@]}"
echo "Seed: $SEED"
echo "=========================================="
echo ""

# Process each split
for SPLIT_NAME in "${SPLITS[@]}"; do
    SPLIT_DIR="$ISLES_BASE/$SPLIT_NAME"

    # Check if split directory exists
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "WARNING: Split directory not found: $SPLIT_DIR"
        echo "Skipping $SPLIT_NAME split..."
        echo ""
        continue
    fi

    echo "=========================================="
    echo "Processing $SPLIT_NAME split"
    echo "=========================================="
    echo ""

    # Extract embeddings for each checkpoint
    echo "[1/4] Extracting embeddings for MCL (Modality Contrastive Learning)..."
    python extract_embeddings_finetune.py \
        --checkpoint "$CHECKPOINT_MCL" \
        --data_dir "$SPLIT_DIR" \
        --output_dir "$OUTPUT_BASE/$SPLIT_NAME/mcl" \
        --modalities $MODALITIES \
        --split full \
        --seed $SEED \
        <<< "yes"  # Auto-confirm the prompt

    echo ""
    echo "[2/4] Extracting embeddings for CL (Contrastive Learning)..."
    python extract_embeddings_finetune.py \
        --checkpoint "$CHECKPOINT_CL" \
        --data_dir "$SPLIT_DIR" \
        --output_dir "$OUTPUT_BASE/$SPLIT_NAME/cl" \
        --modalities $MODALITIES \
        --split full \
        --seed $SEED \
        <<< "yes"

    echo ""
    echo "[3/4] Extracting embeddings for MAE + MCL (Combined)..."
    python extract_embeddings_finetune.py \
        --checkpoint "$CHECKPOINT_MAE_MCL" \
        --data_dir "$SPLIT_DIR" \
        --output_dir "$OUTPUT_BASE/$SPLIT_NAME/mae_mcl" \
        --modalities $MODALITIES \
        --split full \
        --seed $SEED \
        <<< "yes"

    echo ""
    echo "[4/4] Extracting embeddings for MAE + CL (Combined)..."
    python extract_embeddings_finetune.py \
        --checkpoint "$CHECKPOINT_MAE_CL" \
        --data_dir "$SPLIT_DIR" \
        --output_dir "$OUTPUT_BASE/$SPLIT_NAME/mae_cl" \
        --modalities $MODALITIES \
        --split full \
        --seed $SEED \
        <<< "yes"

    echo ""
    echo "✓ Completed $SPLIT_NAME split"
    echo ""
done

echo ""
echo "=========================================="
echo "All extractions complete!"
echo "=========================================="
echo "Results saved in: $OUTPUT_BASE"
echo ""
echo "Output structure:"
for SPLIT_NAME in "${SPLITS[@]}"; do
    if [ -d "$OUTPUT_BASE/$SPLIT_NAME" ]; then
        echo "  $OUTPUT_BASE/$SPLIT_NAME/"
        echo "    ├── mcl/{embeddings.pt, metadata.csv}"
        echo "    ├── cl/{embeddings.pt, metadata.csv}"
        echo "    ├── mae_mcl/{embeddings.pt, metadata.csv}"
        echo "    └── mae_cl/{embeddings.pt, metadata.csv}"
    fi
done
echo "=========================================="
