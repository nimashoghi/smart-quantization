#!/bin/bash

LOGDIR="./lightning_logs_apr28"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1