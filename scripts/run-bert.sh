#!/bin/bash

LOGDIR="./lightning_logs_bert_stsb_mar1"
# rm -rf "$LOGDIR"
# mkdir -p "$LOGDIR"

pwsh ./scripts/train-bert.ps1 --logdir $LOGDIR > "$LOGDIR/output-4.log" 2>&1
