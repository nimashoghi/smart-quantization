# forward + backward
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 4 --max_epochs 16 --compress smart --num_bits_main 6 --num_bits_outlier 8 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[6-8]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress fp8 --name "fp8-[forward,backward,loss,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress s2fp8 --name "s2fp8-[forward,backward,loss,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress fp16 --name "fp16-[forward,backward,loss,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress bf16 --name "bf16-[forward,backward,loss,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress fp32 --name "fp32-[forward,backward,loss,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --num_bits_main 5 --num_bits_outlier 7 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[5-7]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --num_bits_main 4 --num_bits_outlier 6 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[4-6]"
python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --num_bits_main 3 --num_bits_outlier 5 --name "smart-stochastic-[forward,backward,loss,weight,grad,momentum]-[3-5]"

# # forward only
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --no_compress_backward --num_bits_main 6 --num_bits_outlier 8 --name "smart-stochastic-[backward,loss,weight,grad,momentum]-[6-8]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress fp8 --no_compress_backward --name "fp8-[forward,loss,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress s2fp8 --no_compress_backward --name "s2fp8-[forward,loss,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress fp16 --no_compress_backward --name "fp16-[forward,loss,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress bf16 --no_compress_backward --name "bf16-[forward,loss,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --no_compress_backward --num_bits_main 5 --num_bits_outlier 7 --name "smart-stochastic-[backward,loss,weight,grad,momentum]-[5-7]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --no_compress_backward --num_bits_main 4 --num_bits_outlier 6 --name "smart-stochastic-[backward,loss,weight,grad,momentum]-[4-6]"
# python ./train.py --accelerator ddp --gpus -1 --model bert --dataset glue --batch_size 10 --val_batch_size 2 --max_epochs 16 --compress smart --no_compress_backward --num_bits_main 3 --num_bits_outlier 5 --name "smart-stochastic-[backward,loss,weight,grad,momentum]-[3-5]"
