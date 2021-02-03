#!/bin/bash

LOGDIR="./lightning_logs_feb2_bert"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-bert.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
