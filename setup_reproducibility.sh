#!/bin/bash

models=("Analogy" "ComplEx" "ConvE" "DistMult" "HolE" "RESCAL" "RotatE" "TorusE" "TransD" "TransE" "TransH" "TransR" "TuckER")

for model in "${models[@]}"
do
	if [ ! -d /pearlm/pathlm/models/kge_rec/$model/weight_dir_ckpt ]; then
		mkdir pearlm/pathlm/models/kge_rec/$model/weight_dir_ckpt
	fi
done
cp Best\ Checkpoint\ for\ Reproducibility/Analogy_* pearlm/pathlm/models/kge_rec/Analogy/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ComplEx_* pearlm/pathlm/models/kge_rec/ComplEx/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/ConvE_* pearlm/pathlm/models/kge_rec/ConvE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/DistMult_* pearlm/pathlm/models/kge_rec/DistMult/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/HolE_* pearlm/pathlm/models/kge_rec/HolE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RESCAL_* pearlm/pathlm/models/kge_rec/RESCAL/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/RotatE_* pearlm/pathlm/models/kge_rec/RotatE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TorusE_* pearlm/pathlm/models/kge_rec/TorusE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransD_* pearlm/pathlm/models/kge_rec/TransD/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransE_* pearlm/pathlm/models/kge_rec/TransE/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransH_* pearlm/pathlm/models/kge_rec/TransH/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TransR_* pearlm/pathlm/models/kge_rec/TransR/weight_dir_ckpt
cp Best\ Checkpoint\ for\ Reproducibility/TuckER_* pearlm/pathlm/models/kge_rec/TuckER/weight_dir_ckpt

mkdir pearlm/weights
cp UKGCLM\ Weights/* pearlm/weights
echo "[+] All weights have been copied"