#!/bin/bash

# Flushing previous weights
echo "[+] Deleting previous weights"
models=("Analogy" "ComplEx" "ConvE" "DistMult" "HolE" "RESCAL" "RotatE" "TorusE" "TransD" "TransE" "TransH" "TransR" "TuckER")
for model in ${models[@]}
do
  rm -rf kgglm/models/kge/$model/__pycache__
  rm -rf kgglm/models/kge/$model/weight*
  rm -rf kgglm/models/kge/$model/log
  rm kgglm/models/kge/$model/results*
done


if [ "$#" -eq 0 ]
then
  echo "[+] Please specify the GPU Number."
  exit 1
else
  echo "[+] Please make sure you have enough GPU Ram to run the experiments"
  sleep 5
fi

GPU=$1
total_time_start=$(date +%s)

##############################
###### ML1M Experiments ######
##############################

### TransE: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransE"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

### TransH: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransH"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransD: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransD"
dataset="ml1m"
emb_size="100"
batch_size="256"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## TransR: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransR"
dataset="ml1m"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TorusE: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TorusE"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RESCAL: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RESCAL"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### DistMult: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="DistMult"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### ComplEX: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ComplEx"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### Analogy: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="Analogy"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### ConvE: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ConvE"
dataset="ml1m"
emb_size="200"
batch_size="256"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### HolE: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="HolE"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RotatE: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RotatE"
dataset="ml1m"
emb_size="100"
batch_size="256"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TuckER: ml1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TuckER"
dataset="ml1m"
emb_size="100"
batch_size="256"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


#############################
##### LFM1M EXPERIMENTS #####
#############################

### TransE: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransE"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransH: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransH"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransD: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransD"
dataset="lfm1m"
emb_size="100"
batch_size="256"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransR: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransR"
dataset="lfm1m"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TorusE: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TorusE"
dataset="lfm1m"
emb_size="100"
batch_size="256"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RESCAL: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RESCAL"
dataset="lfm1m"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### DistMult: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="DistMult"
dataset="lfm1m"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## ComplEX: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ComplEx"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## Analogy: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="Analogy"
dataset="lfm1m"
emb_size="100"
batch_size="128"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## ConvE: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ConvE"
dataset="lfm1m"
emb_size="200"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

### HolE: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="HolE"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RotatE: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RotatE"
dataset="lfm1m"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## TuckER: lfm1m ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TuckER"
dataset="lfm1m"
emb_size="64"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python kgglm/models/kge/$model/main.py --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > kgglm/models/kge/$model/results_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

# Record the total time elapsed for the execution of the experiments
total_time_end=$(date +%s)
runtime=$((total_time_end-total_time_start))
echo -e "\n\n Total Execution time: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt
