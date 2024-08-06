#!/bin/bash

# Move models and data mapper for gp_baselines
mkdir helper/pathlm/models/kge_rec
mv gp_baselines/mapper_torchkge.py helper/pathlm/data_mappers/.
mv gp_baselines/* helper/pathlm/models/kge_rec/.

# Move the updated r_map for ml1m and lfm1m to the correct folder
cp -f fix/r_map_ml1m.txt helper/data/ml1m/preprocessed/r_map.txt
cp -f fix/r_map_lfm1m.txt helper/data/lfm1m/preprocessed/r_map.txt


touch helper/pathlm/data_mappers/__init__.py
mv fix/kg_test_ml1m.txt helper/data/ml1m/preprocessed/kg_test.txt
mv fix/kg_train_ml1m.txt helper/data/ml1m/preprocessed/kg_train.txt

mv fix/kg_test_lfm1m.txt helper/data/lfm1m/preprocessed/kg_test.txt
mv fix/kg_train_lfm1m.txt helper/data/lfm1m/preprocessed/kg_train.txt

# Fix for kgglm 
mv kgglm helper/pathlm/models/lm/.
mv fix/main_LP.py helper/pathlm/sampling/
mv fix/samplerLP.py helper/pathlm/sampling/samplers/

mv data/lfm1m/* helper/data/lfm1m/
mv data/ml1m/* helper/data/ml1m/
