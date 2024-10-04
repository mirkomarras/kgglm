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


echo -e "\n\n[+] Link Prediction Evaluation\n\n"
# Analogy | ml1m | lfm1m | fb15k-237 | wn18rr
echo "[+] Evaluating Analogy on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/Analogy/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint Analogy_LinkPrediction_dataset_ml1m_ndcg_0.25_mr_1.55_mrr_0.2_hits@1_12.57_hits@3_23.85_hits@10_41.43_epoch_18_e100_bs64_lr0.0001.pth
echo "[+] Evaluating Analogy on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/Analogy/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint Analogy_LinkPrediction_dataset_lfm1m_ndcg_0.54_mr_1.84_mrr_0.5_hits@1_37.26_hits@3_60.88_hits@10_74.95_epoch_3_e100_bs64_lr0.0001.pth
echo "[+] Evaluating Analogy on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/Analogy/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint Analogy_LinkPrediction_dataset_fb15k-237_ndcg_0.47_mr_1.78_mrr_0.41_hits@1_31.25_hits@3_47.19_hits@10_64.67_epoch_11_e100_bs64_lr0.001.pth
echo "[+] Evaluating Analogy on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/Analogy/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint Analogy_LinkPrediction_dataset_wn18rr_ndcg_0.44_mr_0.64_mrr_0.43_hits@1_42.22_hits@3_43.91_hits@10_46.66_epoch_23_e100_bs64_lr0.001.pth



# ComplEx | ml1m | lfm1m | fb15k-237 | wn18rr
echo "[+] Evaluating ComplEx on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ComplEx/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint ComplEx_LinkPrediction_dataset_ml1m_ndcg_0.28_mr_1.57_mrr_0.23_hits@1_14.68_hits@3_26.94_hits@10_44.42_epoch_29_e100_bs64_lr0.0001.pth
echo "[+] Evaluating ComplEx on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ComplEx/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint ComplEx_LinkPrediction_dataset_lfm1m_ndcg_0.69_mr_1.79_mrr_0.68_hits@1_57.77_hits@3_75.07_hits@10_87.59_epoch_2_e100_bs64_lr0.0001.pth
echo "[+] Evaluating ComplEx on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ComplEx/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint ComplEx_LinkPrediction_dataset_fb15k-237_ndcg_0.46_mr_1.71_mrr_0.41_hits@1_30.76_hits@3_46.35_hits@10_62.97_epoch_26_e100_bs64_lr0.0001.pth
echo "[+] Evaluating ComplEx on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ComplEx/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint ComplEx_LinkPrediction_dataset_wn18rr_ndcg_0.43_mr_0.58_mrr_0.42_hits@1_41.07_hits@3_42.79_hits@10_44.84_epoch_19_e100_bs64_lr0.001.pth



# ConvE | ml1m | lfm1m | fb15k-237 | wn18rr
echo "[+] Evaluating ConvE on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ConvE/main.py --lp True --task evaluate --dataset ml1m --embed_size 200 --K 10 --model_checkpoint ConvE_LinkPrediction_dataset_ml1m_ndcg_0.13_mr_0.69_mrr_0.1_hits@1_4.18_hits@3_15.78_hits@10_21.64_epoch_1_e200_bs64_lr0.0001.pth
echo "[+] Evaluating ConvE on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ConvE/main.py --lp True --task evaluate --dataset lfm1m --embed_size 200 --K 10 --model_checkpoint ConvE_LinkPrediction_dataset_lfm1m_ndcg_0.79_mr_1.45_mrr_0.76_hits@1_67.0_hits@3_83.21_hits@10_91.51_epoch_11_e200_bs64_lr0.0001.pth
echo "[+] Evaluating ConvE on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ConvE/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 200 --K 10 --model_checkpoint ConvE_LinkPrediction_dataset_fb15k-237_ndcg_0.1_mr_0.86_mrr_0.07_hits@1_3.31_hits@3_8.66_hits@10_19.57_epoch_12_e200_bs64_lr0.003.pth
#echo "[+] Evaluating ConvE on wn18rr"
#export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/ConvE/main.py --lp True --task evaluate --dataset wn18rr --embed_size 200 --K 10 --model_checkpoint 


# ConvKB | Too Slow

# DistMult | ml1m | ml1m | fb15k-236 | wn18rr
echo "[+] Evaluating DistMult on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/DistMult/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint DistMult_LinkPrediction_dataset_ml1m_ndcg_0.24_mr_1.53_mrr_0.19_hits@1_11.11_hits@3_22.98_hits@10_40.76_epoch_29_e100_bs64_lr0.0001.pth
echo "[+] Evaluating DistMult on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/DistMult/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint DistMult_LinkPrediction_dataset_lfm1m_ndcg_0.24_mr_2.28_mrr_0.17_hits@1_7.13_hits@3_19.85_hits@10_47.89_epoch_14_e100_bs64_lr0.0001.pth
echo "[+] Evaluating DistMult on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/DistMult/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint DistMult_LinkPrediction_dataset_fb15k-237_ndcg_0.42_mr_1.61_mrr_0.37_hits@1_26.71_hits@3_42.92_hits@10_58.28_epoch_27_e100_bs64_lr0.0001.pth
echo "[+] Evaluating DistMult on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/DistMult/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint DistMult_LinkPrediction_dataset_wn18rr_ndcg_0.39_mr_0.48_mrr_0.38_hits@1_37.36_hits@3_38.75_hits@10_39.87_epoch_11_e100_bs64_lr0.001.pth


# HolE | ml1m | lfm1m | fb15k-237 | wn18rr 
echo "[+] Evaluating HolE on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/HolE/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint HolE_LinkPrediction_dataset_ml1m_ndcg_0.17_mr_0.94_mrr_0.14_hits@1_9.78_hits@3_16.09_hits@10_26.56_epoch_24_e100_bs64_lr0.0001.pth
echo "[+] Evaluating HolE on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/HolE/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint HolE_LinkPrediction_dataset_lfm1m_ndcg_0.38_mr_2.13_mrr_0.33_hits@1_22.8_hits@3_36.22_hits@10_60.45_epoch_8_e100_bs64_lr0.0001.pth
echo "[+] Evaluating HolE on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/HolE/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint HolE_LinkPrediction_dataset_fb15k-237_ndcg_0.41_mr_1.73_mrr_0.35_hits@1_24.13_hits@3_41.8_hits@10_58.85_epoch_29_e100_bs256_lr0.001.pth
echo "[+] Evaluating HolE on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/HolE/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint HolE_LinkPrediction_dataset_wn18rr_ndcg_0.38_mr_0.56_mrr_0.37_hits@1_35.9_hits@3_38.82_hits@10_40.67_epoch_29_e100_bs64_lr0.001.pth


# RESCAL | ml1m | lfm1m | fb15k-237 | wn18rr 
echo "[+] Evaluating RESCAL on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RESCAL/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint RESCAL_LinkPrediction_dataset_ml1m_ndcg_0.22_mr_0.81_mrr_0.2_hits@1_16.34_hits@3_20.95_hits@10_29.16_epoch_3_e100_bs64_lr0.0001.pth
echo "[+] Evaluating RESCAL on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RESCAL/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint RESCAL_LinkPrediction_dataset_lfm1m_ndcg_0.33_mr_1.3_mrr_0.29_hits@1_21.25_hits@3_33.62_hits@10_46.26_epoch_3_e100_bs64_lr0.0001.pth
echo "[+] Evaluating RESCAL on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RESCAL/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint RESCAL_LinkPrediction_dataset_fb15k-237_ndcg_0.48_mr_1.58_mrr_0.44_hits@1_34.84_hits@3_48.97_hits@10_63.2_epoch_29_e100_bs256_lr0.001.pth
echo "[+] Evaluating RESCAL on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RESCAL/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint RESCAL_LinkPrediction_dataset_wn18rr_ndcg_0.34_mr_0.72_mrr_0.32_hits@1_28.99_hits@3_34.94_hits@10_39.58_epoch_21_e100_bs64_lr0.001.pth


# RotatE | ml1m | lfm1m | fb15k-237 | wn18rr
echo "[+] Evaluating RotatE on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RotatE/main.py --lp True --task evaluate --dataset ml1m --embed_size 64 --K 10 --model_checkpoint RotatE_LinkPrediction_dataset_ml1m_ndcg_0.28_mr_1.83_mrr_0.22_hits@1_13.12_hits@3_26.26_hits@10_47.5_epoch_12_e64_bs128_lr0.01.pth
echo "[+] Evaluating RotatE on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RotatE/main.py --lp True --task evaluate --dataset lfm1m --embed_size 64 --K 10 --model_checkpoint RotatE_LinkPrediction_dataset_lfm1m_ndcg_0.11_mr_1.57_mrr_0.07_hits@1_1.67_hits@3_6.67_hits@10_26.33_epoch_0_e64_bs128_lr0.01.pth
echo "[+] Evaluating RotatE on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RotatE/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint RotatE_LinkPrediction_dataset_fb15k-237_ndcg_0.49_mr_1.79_mrr_0.43_hits@1_32.16_hits@3_50.39_hits@10_67.2_epoch_29_e100_bs64_lr0.001.pth
echo "[+] Evaluating RotatE on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/RotatE/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint RotatE_LinkPrediction_dataset_wn18rr_ndcg_0.41_mr_0.64_mrr_0.4_hits@1_38.75_hits@3_40.64_hits@10_43.71_epoch_29_e100_bs64_lr0.001.pth


# TorusE | ml1m | lfm1m | fb15k-237 | wn18rr 
echo "[+] Evaluating TorusE on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TorusE/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TorusE_LinkPrediction_dataset_ml1m_ndcg_0.22_mr_1.04_mrr_0.2_hits@1_15.06_hits@3_21.4_hits@10_32.52_epoch_17_e100_bs64_lr0.0001.pth
echo "[+] Evaluating TorusE on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TorusE/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TorusE_LinkPrediction_dataset_lfm1m_ndcg_0.79_mr_1.71_mrr_0.79_hits@1_68.59_hits@3_88.33_hits@10_98.29_epoch_7_e100_bs64_lr0.0001.pth
echo "[+] Evaluating TorusE on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TorusE/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint TorusE_LinkPrediction_dataset_fb15k-237_ndcg_0.42_mr_1.63_mrr_0.37_hits@1_27.26_hits@3_42.17_hits@10_58.02_epoch_27_e100_bs64_lr0.001.pth
echo "[+] Evaluating TorusE on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TorusE/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint TorusE_LinkPrediction_dataset_wn18rr_ndcg_0.21_mr_1.32_mrr_0.15_hits@1_0.99_hits@3_26.17_hits@10_38.62_epoch_20_e100_bs64_lr0.001.pth


# TransD | ml1m | lfm1m | fb15k-237 | wn18rr 
echo "[+] Evaluating TransD on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransD/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransD_LinkPrediction_dataset_ml1m_ndcg_0.21_mr_1.54_mrr_0.16_hits@1_6.58_hits@3_21.34_hits@10_38.9_epoch_29_e100_bs64_lr0.0001.pth
echo "[+] Evaluating TransD on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransD/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransD_LinkPrediction_dataset_lfm1m_ndcg_0.52_mr_3.36_mrr_0.31_hits@1_0.27_hits@3_55.45_hits@10_92.09_epoch_4_e100_bs64_lr0.0001.pth
echo "[+] Evaluating TransD on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransD/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint TransD_LinkPrediction_dataset_fb15k-237_ndcg_0.44_mr_2.01_mrr_0.36_hits@1_20.6_hits@3_48.27_hits@10_67.15_epoch_29_e100_bs64_lr0.001.pth
echo "[+] Evaluating TransD on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransD/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint TransD_LinkPrediction_dataset_wn18rr_ndcg_0.24_mr_1.67_mrr_0.17_hits@1_0.36_hits@3_31.07_hits@10_47.35_epoch_29_e100_bs64_lr0.001.pth


# TransE | ml1m | lfm1m | fb15k-237 | wn18rr
echo "[+] Evaluating TransE on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransE/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransE_LinkPrediction_dataset_ml1m_ndcg_0.22_mr_1.0_mrr_0.19_hits@1_14.45_hits@3_21.3_hits@10_31.74_epoch_23_e100_bs64_lr0.0001.pth
echo "[+] Evaluating TransE on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransE/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransE_LinkPrediction_dataset_lfm1m_ndcg_0.79_mr_2.0_mrr_0.76_hits@1_66.19_hits@3_82.55_hits@10_97.94_epoch_21_e100_bs64_lr0.0001.pth
echo "[+] Evaluating TransE on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransE/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint TransE_LinkPrediction_dataset_fb15k-237_ndcg_0.46_mr_1.66_mrr_0.41_hits@1_31.83_hits@3_46.37_hits@10_62.52_epoch_27_e100_bs64_lr0.001.pth
echo "[+] Evaluating TransE on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransE/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint TransE_LinkPrediction_dataset_wn18rr_ndcg_0.24_mr_1.55_mrr_0.17_hits@1_0.26_hits@3_31.27_hits@10_45.27_epoch_17_e100_bs64_lr0.001.pth


# TransH | ml1m | lfm1m | fb15k-237 | wn18rr
echo -e "\n[+] Evaluating TransH on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransH/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TransH_LinkPrediction_dataset_ml1m_ndcg_0.22_mr_0.99_mrr_0.19_hits@1_14.84_hits@3_20.97_hits@10_31.64_epoch_24_e100_bs64_lr0.0001.pth
echo -e "\n[+] Evaluating TransH on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransH/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TransH_LinkPrediction_dataset_lfm1m_ndcg_0.79_mr_2.07_mrr_0.76_hits@1_66.38_hits@3_82.05_hits@10_97.91_epoch_21_e100_bs64_lr0.0001.pth
echo -e "\n[+] Evaluating TransH on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransH/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint TransH_LinkPrediction_dataset_fb15k-237_ndcg_0.46_mr_1.59_mrr_0.41_hits@1_31.59_hits@3_47.6_hits@10_62.0_epoch_27_e100_bs64_lr0.001.pth
echo -e "\n[+] Evaluating TransH on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransH/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint TransH_LinkPrediction_dataset_wn18rr_ndcg_0.24_mr_1.54_mrr_0.17_hits@1_0.86_hits@3_31.01_hits@10_45.57_epoch_18_e100_bs64_lr0.001.pth


# TransR | ml1m | lfm1m | fb15k-237 | wn18rr
echo "[+] Evaluating TransR on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --lp True --task evaluate --dataset ml1m --embed_size 64 --K 10 --model_checkpoint TransR_LinkPrediction_dataset_ml1m_ndcg_0.17_mr_1.29_mrr_0.13_hits@1_6.36_hits@3_14.94_hits@10_30.43_epoch_29_e64_bs128_lr0.01.pth
echo "[+] Evaluating TransR on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --lp True --task evaluate --dataset lfm1m --embed_size 64 --K 10 --model_checkpoint TransR_LinkPrediction_dataset_lfm1m_ndcg_0.45_mr_2.93_mrr_0.35_hits@1_18.96_hits@3_43.66_hits@10_77.36_epoch_18_e64_bs128_lr0.01.pth
echo "[+] Evaluating TransR on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint TransR_LinkPrediction_dataset_fb15k-237_ndcg_0.27_mr_1.85_mrr_0.19_hits@1_2.8_hits@3_31.63_hits@10_50.11_epoch_27_e100_bs64_lr0.001.pth
echo "[+] Evaluating TransR on wn18rr"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint TransR_LinkPrediction_dataset_wn18rr_ndcg_0.13_mr_0.76_mrr_0.09_hits@1_0.0_hits@3_17.67_hits@10_23.53_epoch_29_e100_bs64_lr0.001.pth


# TuckER | ml1m | lfm1m | fb15k-237 | wn18rr
echo -e "\n[+] Evaluating TuckER on ml1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TransR/main.py --lp True --task evaluate --dataset ml1m --embed_size 100 --K 10 --model_checkpoint TuckER_LinkPrediction_dataset_ml1m_ndcg_0.13_mr_0.65_mrr_0.1_hits@1_4.18_hits@3_15.83_hits@10_21.26_epoch_26_e100_bs256_lr0.0001.pth
echo -e "\n[+] Evaluating TuckER on lfm1m"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --lp True --task evaluate --dataset lfm1m --embed_size 100 --K 10 --model_checkpoint TuckER_LinkPrediction_dataset_lfm1m_ndcg_0.79_mr_1.45_mrr_0.76_hits@1_66.34_hits@3_83.21_hits@10_91.43_epoch_15_e100_bs64_lr0.0001.pth
echo -e "\n[+] Evaluating TuckER on fb15k-237"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --lp True --task evaluate --dataset fb15k-237 --embed_size 100 --K 10 --model_checkpoint TuckER_LinkPrediction_dataset_fb15k-237_ndcg_0.1_mr_0.89_mrr_0.07_hits@1_3.24_hits@3_8.66_hits@10_19.98_epoch_0_e100_bs64_lr0.001.pth
#echo -e "\n[+] Evaluating TuckER on wn18rr"
#export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge_rec/TuckER/main.py --lp True --task evaluate --dataset wn18rr --embed_size 100 --K 10 --model_checkpoint
