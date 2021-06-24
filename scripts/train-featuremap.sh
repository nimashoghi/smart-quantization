#!/bin/bash

LOGDIR="./lightning_logs_jun23_resnet_fmaponly"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-featuremaponly.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
