python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress s2fp8 --name "s2fp8-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress smart --num_bits_main 6 --num_bits_outlier 8 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[6-8]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress fp32 --name "fp32-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress fp8 --name "fp8-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress fp16 --name "fp16-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress bf16 --name "bf16-[forward,backward,loss,weight,grad,momentum]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress smart --num_bits_main 5 --num_bits_outlier 7 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[5-7]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress smart --num_bits_main 4 --num_bits_outlier 6 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[4-6]" @args
python ./train.py --accelerator ddp --gpus -1 --model resnet --dataset cifar10 --batch_size 256 --val_batch_size 256 --max_epochs 384 --scheduler_type multi_step --compress smart --num_bits_main 3 --num_bits_outlier 5 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[3-5]" @args