#!/bin/bash

LOGDIR="./lightning_logs_feb2_test"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-resnet-new.ps1 --logdir $LOGDIR > "$LOGDIR/output-new.log" 2>&1
