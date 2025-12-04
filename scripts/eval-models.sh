#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset> [optional additional arguments e.g. --lr 0.001 --epochs 100]"
    exit 1
fi

DATASET=$1
shift   # Remove the first argument so that $@ contains only the extra arguments
EXTRA_ARGS="$@"

BACKBONES=("GAT" "GCN" "GIN")
MODES=("T" "S" "P")

for backbone in "${BACKBONES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "======================================="
        echo "Starting: backbone=$backbone | train_mode=$mode"
        echo "---------------------------------------"

        python main.py \
            --dataset "$DATASET" \
            --backbone "$backbone" \
            --train_mode "$mode" \
            $EXTRA_ARGS
    done
done
