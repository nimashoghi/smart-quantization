# forward + backward
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --num_bits_main 6 --num_bits_outlier 8 --name "smart-stochastic-[forward,backward,weight,grad,momentum]-[6-8]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress FP8 --name "fp8-[forward,backward,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress S2FP8 --name "s2fp8-[forward,backward,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress FP16 --name "fp16-[forward,backward,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress BF16 --name "bf16-[forward,backward,weight,grad,momentum]"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --num_bits_main 5 --num_bits_outlier 7 --name "smart-stochastic-[forward,backward,weight,grad,momentum]-[5-7]"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --num_bits_main 4 --num_bits_outlier 6 --name "smart-stochastic-[forward,backward,weight,grad,momentum]-[4-6]"
python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --num_bits_main 3 --num_bits_outlier 5 --name "smart-stochastic-[forward,backward,weight,grad,momentum]-[3-5]"

# # forward only
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --no_compress_backward --num_bits_main 6 --num_bits_outlier 8 --name "smart-stochastic-[backward,weight,grad,momentum]-[6-8]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress FP8 --no_compress_backward --name "fp8-[forward,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress S2FP8 --no_compress_backward --name "s2fp8-[forward,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress FP16 --no_compress_backward --name "fp16-[forward,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress BF16 --no_compress_backward --name "bf16-[forward,weight,grad,momentum]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --no_compress_backward --num_bits_main 5 --num_bits_outlier 7 --name "smart-stochastic-[backward,weight,grad,momentum]-[5-7]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --no_compress_backward --num_bits_main 4 --num_bits_outlier 6 --name "smart-stochastic-[backward,weight,grad,momentum]-[4-6]"
# python ./train.py --accelerator ddp --gpus -1 --batch_size 8192 --max_epochs 128 --compress SmartCompress --no_compress_backward --num_bits_main 3 --num_bits_outlier 5 --name "smart-stochastic-[backward,weight,grad,momentum]-[3-5]"
