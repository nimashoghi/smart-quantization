#!/bin/bash

LOGDIR="./lightning_logs_jun18_inception"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

CUDA_VISIBLE_DEVICES=1 pwsh ./scripts/train.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
