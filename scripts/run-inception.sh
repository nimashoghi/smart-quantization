#!/bin/bash

LOGDIR="./lightning_logs_feb6_inception"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-inception.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
