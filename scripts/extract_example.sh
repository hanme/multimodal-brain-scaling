#!/usr/bin/env bash
# End-to-end feature extraction example.

set -euo pipefail

MODEL_ID="${MODEL_ID:-resnet50_imagenet_full}"
DATA_ROOT="${DATA_ROOT:-./data/stimuli/object_images}"
DATASET_TYPE="${DATASET_TYPE:-h5}"
STIMULUS_SET_ID="${STIMULUS_SET_ID:-object_images}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/features/${MODEL_ID}}"
COMMITTED_EXTRACTION_LAYERS="${COMMITTED_EXTRACTION_LAYERS:-configs/evaluation/layer_commitment/committed_extraction_layers.json}"

uv run --extra evaluation mbs-extract-features \
    --model_id "$MODEL_ID" \
    --backbone_source spvvs \
    --data_root "$DATA_ROOT" \
    --dataset_type "$DATASET_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --committed_extraction_layers "$COMMITTED_EXTRACTION_LAYERS" \
    --stimulus_set_id "$STIMULUS_SET_ID" \
    --batch_size 32 \
    --num_workers 4
