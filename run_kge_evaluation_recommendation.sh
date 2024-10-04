#!/bin/bash

if [ "$#" -eq 0 ]
then
  echo "[+] Please specify the GPU Number."
  exit 1
else
  echo "[+] Please make sure you have enough GPU Ram to run the evaluation."
  sleep 5
fi

GPU=$1

# Analogy | ml1m | lfm1m

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/Analogy/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint Analogy_dataset_ml1m_ndcg_0.26_mrr_0.21_prec_0.1_rec_0.03_ser_0.29_div_0.33_nov_0.93_cov_0.06_epoch_4_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/Analogy/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint Analogy_dataset_lfm1m_ndcg_0.33_mrr_0.28_prec_0.12_rec_0.03_ser_0.98_div_0.48_nov_0.87_cov_0.45_epoch_27_e100_bs128_lr0.0001.pth

# ComplEx | ml1m | lfm1m

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ComplEx/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint ComplEx_dataset_ml1m_ndcg_0.26_mrr_0.21_prec_0.1_rec_0.03_ser_0.3_div_0.37_nov_0.93_cov_0.05_epoch_2_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint ComplEx_dataset_lfm1m_ndcg_0.28_mrr_0.23_prec_0.1_rec_0.03_ser_0.97_div_0.52_nov_0.87_cov_0.43_epoch_29_e100_bs64_lr0.0001.pth

# ConvE | ml1m | lfm1m

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ConvE/main.py --task evaluate --dataset ml1m --embed_size 200 --K 10 --model_checkpoint ConvE_dataset_ml1m_ndcg_0.27_mrr_0.22_prec_0.1_rec_0.04_ser_0.22_div_0.38_nov_0.93_cov_0.04_epoch_2_e200_bs256_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint ConvE_dataset_lfm1m_ndcg_0.13_mrr_0.11_prec_0.03_rec_0.01_ser_0.47_div_0.58_nov_0.88_cov_0.0_epoch_2_e200_bs64_lr0.0001.pth

# ConvKB | Too Slow

# DistMult | ml1m | ml1m 
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/DistMult/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint DistMult_dataset_ml1m_ndcg_0.28_mrr_0.22_prec_0.11_rec_0.04_ser_0.32_div_0.4_nov_0.93_cov_0.07_epoch_22_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/DistMult/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint DistMult_dataset_lfm1m_ndcg_0.3_mrr_0.25_prec_0.11_rec_0.03_ser_0.97_div_0.56_nov_0.87_cov_0.35_epoch_10_e100_bs256_lr0.001.pth

# HolE | ml1m | lfm1m 

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/HolE/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint HolE_dataset_ml1m_ndcg_0.24_mrr_0.19_prec_0.09_rec_0.03_ser_0.5_div_0.39_nov_0.93_cov_0.14_epoch_11_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/HolE/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint HolE_dataset_lfm1m_ndcg_0.19_mrr_0.14_prec_0.06_rec_0.02_ser_0.96_div_0.55_nov_0.87_cov_0.27_epoch_29_e100_bs64_lr0.0001.pth

# RESCAL | ml1m | lfm1m 

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RESCAL/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint RESCAL_dataset_ml1m_ndcg_0.26_mrr_0.2_prec_0.09_rec_0.03_ser_0.42_div_0.41_nov_0.92_cov_0.19_epoch_3_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RESCAL/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint RESCAL_dataset_lfm1m_ndcg_0.24_mrr_0.19_prec_0.08_rec_0.02_ser_0.99_div_0.61_nov_0.88_cov_0.54_epoch_26_e100_bs256_lr0.001.pth

# RotatE | ml1m | lfm1m 
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RotatE/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint RotatE_dataset_ml1m_ndcg_0.2_mrr_0.15_prec_0.07_rec_0.02_ser_0.76_div_0.49_nov_0.93_cov_0.4_epoch_23_e100_bs256_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RotatE/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint RotatE_dataset_lfm1m_ndcg_0.25_mrr_0.2_prec_0.09_rec_0.02_ser_0.97_div_0.52_nov_0.87_cov_0.42_epoch_29_e100_bs256_lr0.001.pth

# TorusE | ml1m | lfm1m 

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TorusE/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TorusE_dataset_ml1m_ndcg_0.26_mrr_0.21_prec_0.08_rec_0.03_ser_0.61_div_0.43_nov_0.88_cov_0.04_epoch_0_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TorusE/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TorusE_dataset_lfm1m_ndcg_0.18_mrr_0.14_prec_0.05_rec_0.01_ser_0.9_div_0.62_nov_0.86_cov_0.3_epoch_28_e100_bs256_lr0.0001.pth

# TransD | ml1m | lfm1m 

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransD/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransD_dataset_ml1m_ndcg_0.23_mrr_0.18_prec_0.08_rec_0.03_ser_0.84_div_0.42_nov_0.94_cov_0.48_epoch_27_e100_bs256_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransD/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransD_dataset_lfm1m_ndcg_0.17_mrr_0.13_prec_0.06_rec_0.01_ser_0.83_div_0.55_nov_0.87_cov_0.3_epoch_29_e100_bs256_lr0.0001.pth

# TransE | ml1m | lfm1m

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransE/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransE_dataset_ml1m_ndcg_0.28_mrr_0.23_prec_0.1_rec_0.03_ser_0.33_div_0.41_nov_0.93_cov_0.04_epoch_16_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransE/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransE_dataset_lfm1m_ndcg_0.12_mrr_0.1_prec_0.03_rec_0.01_ser_0.59_div_0.54_nov_0.85_cov_0.0_epoch_28_e100_bs64_lr0.001.pth

# TransH | ml1m | lfm1m
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransH/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransH_dataset_ml1m_ndcg_0.26_mrr_0.2_prec_0.09_rec_0.03_ser_0.43_div_0.45_nov_0.93_cov_0.14_epoch_10_e100_bs64_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransH/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransH_dataset_lfm1m_ndcg_0.18_mrr_0.14_prec_0.05_rec_0.01_ser_0.99_div_0.66_nov_0.88_cov_0.36_epoch_22_e100_bs64_lr0.001.pth

# # TransR | ml1m | lfm1m 
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransR_dataset_ml1m_ndcg_0.16_mrr_0.12_prec_0.05_rec_0.02_ser_0.85_div_0.47_nov_0.93_cov_0.95_epoch_2_e100_bs256_lr0.001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransR_dataset_lfm1m_ndcg_0.15_mrr_0.11_prec_0.04_rec_0.01_ser_1.0_div_0.7_nov_0.9_cov_0.7_epoch_29_e100_bs256_lr0.001.pth

# TuckER | ml1m | lfm1m
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TuckER_dataset_ml1m_ndcg_0.27_mrr_0.22_prec_0.1_rec_0.04_ser_0.16_div_0.36_nov_0.93_cov_0.04_epoch_19_e100_bs256_lr0.0001.pth

export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TuckER_dataset_lfm1m_ndcg_0.12_mrr_0.1_prec_0.03_rec_0.01_ser_0.32_div_0.29_nov_0.83_cov_0.02_epoch_4_e64_bs64_lr0.0001.pth


