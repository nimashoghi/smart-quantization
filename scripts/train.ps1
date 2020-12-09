python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128  --compress NoCompression --name "fp32"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128  --compress SmartCompress --name "smart"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128  --compress FP8 --name "fp8"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128  --compress S2FP8 --name "s2fp8"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128  --compress FP16 --name "fp16"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128  --compress BF16 --name "bf16"


python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress_weights --compress_gradients --compress_momentum_vectors --compress NoCompression --name "fp32-opt-compress"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress_weights --compress_gradients --compress_momentum_vectors --compress SmartCompress --name "smart-opt-compress"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress_weights --compress_gradients --compress_momentum_vectors --compress FP8 --name "fp8-opt-compress"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress_weights --compress_gradients --compress_momentum_vectors --compress S2FP8 --name "s2fp8-opt-compress"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress_weights --compress_gradients --compress_momentum_vectors --compress FP16 --name "fp16-opt-compress"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress_weights --compress_gradients --compress_momentum_vectors --compress BF16 --name "bf16-opt-compress"
