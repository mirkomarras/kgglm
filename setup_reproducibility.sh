#!/bin/bash

models=("Analogy" "ComplEx" "ConvE" "DistMult" "HolE" "RESCAL" "RotatE" "TorusE" "TransD" "TransE" "TransH" "TransR" "TuckER")

for model in "${models[@]}"
do
	if [ ! -d /helper/models/kge/$model/weight_dir_ckpt ]; then
		mkdir helper/models/kge/$model/weight_dir_ckpt
	fi
done
cp Best\ Checkpoint\ for\ Reproducibility/Analogy_* helper/models/kge/Analogy/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ComplEx_* helper/models/kge/ComplEx/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ConvE_* helper/models/kge/ConvE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/DistMult_* helper/models/kge/DistMult/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/HolE_* helper/models/kge/HolE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RESCAL_* helper/models/kge/RESCAL/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RotatE_* helper/models/kge/RotatE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TorusE_* helper/models/kge/TorusE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransD_* helper/models/kge/TransD/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransE_* helper/models/kge/TransE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransH_* helper/models/kge/TransH/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransR_* helper/models/kge/TransR/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TuckER_* helper/models/kge/TuckER/weight_dir_ckpt

mkdir helper/weights
cp UKGCLM\ Weights/* helper/weights
echo "[+] All weights have been copied"