#!/bin/bash

LOGDIR="./lightning_logs_inception_feb1"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-inception.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
