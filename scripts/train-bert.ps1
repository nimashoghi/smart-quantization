# forward + backward
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 6 --num_bits_outlier 8 --measure_compression_ratio @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress fp8 @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress s2fp8 --measure_compression_ratio @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress fp16 @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress bf16 @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress fp32 @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 5 --num_bits_outlier 7 --measure_compression_ratio @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 4 --num_bits_outlier 6 --measure_compression_ratio @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 3 --num_bits_outlier 5 --measure_compression_ratio @args

python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 6 --num_bits_outlier 8 --measure_compression_ratio --no_compress_momentum_vectors --no_compress_backward @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress s2fp8 --measure_compression_ratio --no_compress_momentum_vectors --no_compress_backward @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 5 --num_bits_outlier 7 --measure_compression_ratio --no_compress_momentum_vectors --no_compress_backward @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 4 --num_bits_outlier 6 --measure_compression_ratio --no_compress_momentum_vectors --no_compress_backward @args
python ./train.py --git --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 100 --compress smart --num_bits_main 3 --num_bits_outlier 5 --measure_compression_ratio --no_compress_momentum_vectors --no_compress_backward @args
