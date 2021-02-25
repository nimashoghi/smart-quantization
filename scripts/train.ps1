python ./train.py --model bert --dataset glue --task_name rte --batch_size 10 --val_batch_size 2 --max_epochs 5 --optimizer adamw --learning_rate 1e-5 --weight_decay 0.0 --epsilon 1e-8 @args
