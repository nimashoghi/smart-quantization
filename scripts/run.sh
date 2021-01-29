#!/bin/bash

LOGDIR="./lightning_logs_resnet_fixed_jan29"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-resnet.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
