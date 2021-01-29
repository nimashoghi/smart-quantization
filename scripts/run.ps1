$logdir = "./lightning_logs_resnet_fixed_jan29"
rm -ErrorAction Ignore -Recurse -Force "$logdir"
mkdir -ErrorAction Ignore "$logdir"
./scripts/train-resnet.ps1 --logdir "$logdir" | Tee-Object $logdir/output.log
