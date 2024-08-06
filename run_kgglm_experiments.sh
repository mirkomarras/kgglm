#!/bin/bash
# All the models must be run from the helper/ directory
if [[ "$(pwd)" != "$(pwd)/helper" ]]; then
  cd helper/
fi


echo -e "\n\n Pretrain LFM1M\n\n"
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/helper_main.py --task pretrain --dataset lfm1m --sample_size 500 --model distilgpt2 --nproc 8 --n_hop 5 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 3 --validation_interval 1
echo -e "\n\nFinetuning LFM1M LP 250 1\n\n"
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/tokenize_dataset.py --dataset lfm1m --task finetuneLP --sample_size 250 --n_hop 1
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/helper_main.py --task finetuneLP --dataset lfm1m --sample_size 250 --model distilgpt2 --nproc 8 --n_hop 1 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 687 --pretrain_ckpt helper/weights/lfm1m/pretrain@lfm1m@distilgpt2@500@5@gcd
echo -e "\n\nFinetuning LFM1M Rec 250 3\n\n"
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/tokenize_dataset.py --dataset lfm1m --task finetuneRec --sample_size 250 --n_hop 3
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/helper_main.py --task finetuneRec --dataset lfm1m --sample_size 250 --model distilgpt2 - nproc 8 --n_hop 3 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 4658 --pretrain_ckpt helper/weights/lfm1m/pretrain@lfm1m@distilgpt2@500@5@gcd

echo -e "\n\n Pretrain ML1M\n\n"
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/helper_main.py --task pretrain --dataset ml1m --sample_size 500 --model distilgpt2 --nproc 8 --n_hop 5 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 3 --validation_interval 12364
echo -e "\n\nFinetuning ML1M LP 250 1\n\n"
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/tokenize_dataset.py --dataset ml1m --task finetuneLP --sample_size 250 --n_hop 1
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/helper_main.py --task finetuneLP --dataset ml1m --sample_size 250 --model distilgpt2 --nproc 8 --n_hop 1 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 907 --pretrain_ckpt helper/weights/ml1m/pretrain@ml1m@distilgpt2@500@5@gcd
echo -e "\n\nFinetuning ML1M Rec 250 3\n\n"
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/tokenize_dataset.py --dataset ml1m --task finetuneRec --sample_size 250 --n_hop 3
export CUDA_VISIBLE_DEVICES=1 && python pathlm/models/lm/helper/helper_main.py --task finetuneRec --dataset ml1m --sample_size 250 --model distilgpt2 --nproc 8 --n_hop 3 --batch_size 256 --infer_batch_size 128 --eval_device cuda:0 --logit_processor_type gcd --num_epochs 5 --validation_interval 5895 --pretrain_ckpt helper/weights/ml1m/pretrain@ml1m@distilgpt2@500@5@gcd


