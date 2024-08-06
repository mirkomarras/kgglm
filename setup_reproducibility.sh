#!/bin/bash

models=("Analogy" "ComplEx" "ConvE" "DistMult" "HolE" "RESCAL" "RotatE" "TorusE" "TransD" "TransE" "TransH" "TransR" "TuckER")

for model in "${models[@]}"
do
	if [ ! -d /helper/pathlm/models/kge_rec/$model/weight_dir_ckpt ]; then
		mkdir helper/pathlm/models/kge_rec/$model/weight_dir_ckpt
	fi
done
cp Best\ Checkpoint\ for\ Reproducibility/Analogy_* helper/pathlm/models/kge_rec/Analogy/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ComplEx_* helper/pathlm/models/kge_rec/ComplEx/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ConvE_* helper/pathlm/models/kge_rec/ConvE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/DistMult_* helper/pathlm/models/kge_rec/DistMult/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/HolE_* helper/pathlm/models/kge_rec/HolE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RESCAL_* helper/pathlm/models/kge_rec/RESCAL/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RotatE_* helper/pathlm/models/kge_rec/RotatE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TorusE_* helper/pathlm/models/kge_rec/TorusE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransD_* helper/pathlm/models/kge_rec/TransD/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransE_* helper/pathlm/models/kge_rec/TransE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransH_* helper/pathlm/models/kge_rec/TransH/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransR_* helper/pathlm/models/kge_rec/TransR/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TuckER_* helper/pathlm/models/kge_rec/TuckER/weight_dir_ckpt

mkdir helper/weights
cp UKGCLM\ Weights/* helper/weights
echo "[+] All weights have been copied"