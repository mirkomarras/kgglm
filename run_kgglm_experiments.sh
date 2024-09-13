#!/bin/bash

echo -e "\n\n Pretrain LFM1M\n\n"
echo -e "\n[+] Tokenizing pretrain dataset" 
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/tokenize_dataset.py --dataset lfm1m --task pretrain --sample_size 500 --n_hop 5 --train_tokenizer True
echo -e "\n[+] Training"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/kgglm_main.py --task pretrain --dataset lfm1m --sample_size 500 --model distilgpt2 --nproc 8 --n_hop 5 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 3 --validation_interval 1 #23992
echo -e "\n\nFinetuning LFM1M LP 250 1\n\n"
echo -e "\n[+] Tokenizing finetuning lp dataset" 
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/tokenize_dataset.py --dataset lfm1m --task finetuneLP --sample_size 250 --n_hop 1
echo -e "\n[+] Training"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/kgglm_main.py --task finetuneLP --dataset lfm1m --sample_size 250 --model distilgpt2 --nproc 8 --n_hop 1 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 687 --pretrain_ckpt helper/weights/lfm1m/pretrain@lfm1m@distilgpt2@500@5@gcd
echo -e "\n\nFinetuning LFM1M Rec 250 3\n\n"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/tokenize_dataset.py --dataset lfm1m --task finetuneRec --sample_size 250 --n_hop 3
echo -e "\n[+] Training"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/kgglm_main.py --task finetuneRec --dataset lfm1m --sample_size 250 --model distilgpt2 - nproc 8 --n_hop 3 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 4658 --pretrain_ckpt helper/weights/lfm1m/pretrain@lfm1m@distilgpt2@500@5@gcd

echo -e "\n\n Pretrain ML1M\n\n"
echo -e "\n[+] Tokenizing pretrain dataset" 
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/tokenize_dataset.py --dataset lfm1m --task pretrain --sample_size 500 --n_hop 5 --train_tokenizer True
echo -e "\n[+] Training"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/kgglm_main.py --task pretrain --dataset ml1m --sample_size 500 --model distilgpt2 --nproc 8 --n_hop 5 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 3 --validation_interval 12364
echo -e "\n\nFinetuning ML1M LP 250 1\n\n"
echo -e "\n[+] Tokenizing finetuning lp dataset" 
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/tokenize_dataset.py --dataset ml1m --task finetuneLP --sample_size 250 --n_hop 1
echo -e "\n[+] Training"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/kgglm_main.py --task finetuneLP --dataset ml1m --sample_size 250 --model distilgpt2 --nproc 8 --n_hop 1 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 907 --pretrain_ckpt helper/weights/ml1m/pretrain@ml1m@distilgpt2@500@5@gcd
echo -e "\n\nFinetuning ML1M Rec 250 3\n\n"
echo -e "\n[+] Tokenizing finetuning recommendation dataset" 
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/tokenize_dataset.py --dataset ml1m --task finetuneRec --sample_size 250 --n_hop 3
echo -e "\n[+] Training"
export CUDA_VISIBLE_DEVICES=1 && python helper/models/lm/KGGLM/kgglm_main.py --task finetuneRec --dataset ml1m --sample_size 250 --model distilgpt2 --nproc 8 --n_hop 3 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 5895 --pretrain_ckpt helper/weights/ml1m/pretrain@ml1m@distilgpt2@500@5@gcd


