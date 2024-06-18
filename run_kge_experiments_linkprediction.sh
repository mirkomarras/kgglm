#!/bin/bash

# All the models must be run from the pearlm/ directory
if [[ "$(pwd)" != "$(pwd)/pearlm" ]]; then
  cd pearlm/
fi


### Flushing previous weights
echo "[+] Deleting previous weights"
models=("Analogy" "ComplEx" "ConvE" "DistMult" "HolE" "RESCAL" "RotatE" "TorusE" "TransD" "TransE" "TransH" "TransR" "TuckER")
for model in ${models[@]}
do
  rm -rf pathlm/models/kge_rec/$model/__pycache__
  rm -rf pathlm/models/kge_rec/$model/weight*
  rm -rf pathlm/models/kge_rec/$model/log
  rm pathlm/models/kge_rec/$model/results*
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

###################################
###### FB15k-237 Experiments ######
###################################

### TransE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransE"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

### TransH ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransH"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransD ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransD"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## TransR ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransR"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TorusE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TorusE"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RESCAL ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RESCAL"
dataset="FB15k-237"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### DistMult ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="DistMult"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### ComplEX ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ComplEx"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### Analogy ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="Analogy"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### ConvE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ConvE"
dataset="FB15k-237"
emb_size="200"
batch_size="64"
lr="0.003"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### HolE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="HolE"
dataset="FB15k-237"
emb_size="100"
batch_size="256"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RotatE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RotatE"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TuckER ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TuckER"
dataset="FB15k-237"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


#############################
##### WN18RR EXPERIMENTS #####
#############################

### TransE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransE"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransH ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransH"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransD ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransD"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransR ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransR"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TorusE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TorusE"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RESCAL ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RESCAL"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### DistMult ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="DistMult"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## ComplEX ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ComplEx"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## Analogy ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="Analogy"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## ConvE ###
# start=$(date +%s)
# echo -e "\n\t\t$(date)\n"
# model="ConvE"
# dataset="wn18rr"
# emb_size="200"
# batch_size="64"
# lr="0.0001"
# wd="0"
# k="10"
# use_cuda="all"
# echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
# export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

### HolE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="HolE"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RotatE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RotatE"
dataset="wn18rr"
emb_size="100"
batch_size="64"
lr="0.001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## TuckER ###
# start=$(date +%s)
# echo -e "\n\t\t$(date)\n"
# model="TuckER"
# dataset="wn18rr"
# emb_size="64"
# batch_size="64"
# lr="0.0001"
# wd="0"
# k="10"
# use_cuda="all"
# echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
# export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

#############################
##### ml1m EXPERIMENTS #####
#############################

### TransE ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransH ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransD ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransD"
dataset="ml1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransR ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransR"
dataset="ml1m"
emb_size="64"
batch_size="128"
lr="0.01"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TorusE ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RESCAL ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### DistMult ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## ComplEX ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## Analogy ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## ConvE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="ConvE"
dataset="ml1m"
emb_size="200"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

### HolE ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RotatE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RotatE"
dataset="ml1m"
emb_size="64"
batch_size="128"
lr="0.01"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## TuckER ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

#############################
##### lfm1m EXPERIMENTS #####
#############################

### TransE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransE"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransH ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransH"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransD ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransD"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TransR ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TransR"
dataset="lfm1m"
emb_size="64"
batch_size="128"
lr="0.01"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### TorusE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TorusE"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RESCAL ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RESCAL"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### DistMult ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="DistMult"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
margin=1
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## ComplEX ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## Analogy ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="Analogy"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

## ConvE ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

### HolE ###
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
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda --margin $margin > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


### RotatE ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="RotatE"
dataset="lfm1m"
emb_size="64"
batch_size="128"
lr="0.01"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt


## TuckER ###
start=$(date +%s)
echo -e "\n\t\t$(date)\n"
model="TuckER"
dataset="lfm1m"
emb_size="100"
batch_size="64"
lr="0.0001"
wd="0"
k="10"
use_cuda="all"
echo -e "\n\n\t\t Model: $model | dataset: $dataset | Emb. Size: $emb_size| Batch Size: $batch_size | Learning Rate -> $lr \n\n"
export CUDA_VISIBLE_DEVICES=$GPU && python pathlm/models/kge_rec/$model/main.py --lp True --dataset $dataset --epoch 30 --embed_size $emb_size --batch_size $batch_size --weight_decay $wd --lr $lr --K $k --use_cuda $use_cuda > pathlm/models/kge_rec/$model/results_linkprediction_$dataset.txt ; end=$(date +%s) ; runtime=$((end-start)) ; echo "Execution time $model $dataset: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt

# Record the total time elapsed for the execution of the experiments
total_time_end=$(date +%s)
runtime=$((total_time_end-total_time_start))
echo -e "\n\n Total Execution time: $((runtime / 3600)) hours $(((runtime / 60) % 60)) minutes $((runtime % 60)) seconds" >> elapsed_training.txt
