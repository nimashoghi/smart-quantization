#!/bin/bash

LOGDIR="./lightning_logs_apr25"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train.ps1 --logdir $LOGDIR > "$LOGDIR/output-new.log" 2>&1
