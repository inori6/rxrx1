#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
    pwd
)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="configs/final_selection/02_aggressive_tta_final.yaml"

mkdir -p outputs/submissions outputs/logs

declare -a RUNS=(
"50 0.5122"
"54 0.5187"
"59 0.5318"
)

for item in "${RUNS[@]}"; do
    read EPOCH ACC <<< "$item"

    CKPT="outputs/checkpoints/final_select_aggressive/model_snapshots/epoch_${EPOCH}_trainacc_${ACC}.pt"

    TTA="outputs/submissions/aggressive_epoch${EPOCH}_tta_trainnorm.csv"
    LOGITS="outputs/submissions/aggressive_epoch${EPOCH}_tta_logits.pkl"
    TTALSA="outputs/submissions/aggressive_epoch${EPOCH}_tta_lsa_trainnorm.csv"

    echo
    echo "================================================================"
    echo "EPOCH ${EPOCH} | TTA"
    echo "checkpoint = ${CKPT}"
    echo "================================================================"


    uv run python -u scripts/predict_test_submit.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --output "$TTA" \
        --logits-output "$LOGITS" \
        --norm-stats-source train \
        2>&1 | tee "outputs/logs/epoch${EPOCH}_tta.log"

    echo
    echo "================================================================"
    echo "EPOCH ${EPOCH} | TTA -> LSA"
    echo "NO GPU FORWARD / REUSE CACHED LOGITS"
    echo "================================================================"

    uv run python -u scripts/apply_lsa_from_logits.py \
        --config "$CONFIG" \
        --logits "$LOGITS" \
        --output "$TTALSA" \
        2>&1 | tee "outputs/logs/epoch${EPOCH}_tta_lsa.log"

    echo
    echo "Finished epoch ${EPOCH}"
    ls -lh "$TTA" "$LOGITS" "$TTALSA"
done


echo
echo "================================================================"
echo "ALL DONE"
echo "================================================================"

ls -lh \
outputs/submissions/aggressive_epoch50_tta_trainnorm.csv \
outputs/submissions/aggressive_epoch50_tta_lsa_trainnorm.csv \
outputs/submissions/aggressive_epoch54_tta_trainnorm.csv \
outputs/submissions/aggressive_epoch54_tta_lsa_trainnorm.csv \
outputs/submissions/aggressive_epoch59_tta_trainnorm.csv \
outputs/submissions/aggressive_epoch59_tta_lsa_trainnorm.csv
