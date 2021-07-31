#!/bin/bash

LOGDIR="./lightning_logs_jul31_resnet_perf"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
