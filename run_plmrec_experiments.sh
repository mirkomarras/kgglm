#!/bin/bash

echo -e "\n\n Training PLM-Rec on ML1M\n\n"
export CUDA_VISIBLE_DEVICES=1 && python kgglm/models/lm/PLM/tokenize_dataset.py --dataset ml1m --task end-to-end --sample_size 500 --nproc 2 --n_hop 5
export CUDA_VISIBLE_DEVICES=1 && python kgglm/models/lm/PLM/main.py --validation_interval 5895 --num_epochs 20 --logit_processor_type plm --dataset ml1m

echo -e "\n\n Training PLM-Rec on LFM1M\n\n"
export CUDA_VISIBLE_DEVICES=1 && python kgglm/models/lm/PLM/tokenize_dataset.py --dataset lfm1m --task end-to-end --sample_size 500 --nproc 2 --n_hop 5
export CUDA_VISIBLE_DEVICES=1 && python kgglm/models/lm/PLM/main.py --validation_interval 4658 --num_epochs 20 --logit_processor_type plm --dataset lfm1m