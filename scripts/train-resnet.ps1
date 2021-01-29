python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress s2fp8 --name "s2fp8-[forward,backward,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress s2fp8 --no_compress_momentum_vectors --name "s2fp8-[forward,backward,weight,grad]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress smart --num_bits_main 6 --num_bits_outlier 8 --measure_compression_ratio --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[6-8]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress fp32 --name "fp32-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress fp8 --name "fp8-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress fp16 --name "fp16-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress bf16 --name "bf16-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress smart --num_bits_main 5 --num_bits_outlier 7 --measure_compression_ratio --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[5-7]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress smart --num_bits_main 4 --num_bits_outlier 6 --measure_compression_ratio --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[4-6]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 128 --val_batch_size 64 --max_epochs 384 --scheduler_type multi_step --scheduler_gamma 0.1 --scheduler_milestones 100 150 200 --compress smart --num_bits_main 3 --num_bits_outlier 5 --measure_compression_ratio --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[3-5]" @args
