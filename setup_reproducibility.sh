#!/bin/bash

models=("Analogy" "ComplEx" "ConvE" "DistMult" "HolE" "RESCAL" "RotatE" "TorusE" "TransD" "TransE" "TransH" "TransR" "TuckER")

for model in "${models[@]}"
do
	if [ ! -d /kgglm/models/kge/$model/weight_dir_ckpt ]; then
		mkdir kgglm/models/kge/$model/weight_dir_ckpt
	fi
done
cp Best\ Checkpoint\ for\ Reproducibility/Analogy_* kgglm/models/kge/Analogy/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ComplEx_* kgglm/models/kge/ComplEx/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ConvE_* kgglm/models/kge/ConvE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/DistMult_* kgglm/models/kge/DistMult/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/HolE_* kgglm/models/kge/HolE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RESCAL_* kgglm/models/kge/RESCAL/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RotatE_* kgglm/models/kge/RotatE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TorusE_* kgglm/models/kge/TorusE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransD_* kgglm/models/kge/TransD/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransE_* kgglm/models/kge/TransE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransH_* kgglm/models/kge/TransH/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransR_* kgglm/models/kge/TransR/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TuckER_* kgglm/models/kge/TuckER/weight_dir_ckpt

mkdir kgglm/weights
cp UKGCLM\ Weights/* kgglm/weights
echo "[+] All weights have been copied"