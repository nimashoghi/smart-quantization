#!/bin/bash

LOGDIR="./lightning_logs_imdb_feb23"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

pwsh ./scripts/train-bert-imdb.ps1 --logdir $LOGDIR > "$LOGDIR/output.log" 2>&1
