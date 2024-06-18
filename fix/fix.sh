#!/bin/bash

# Move models and data mapper for KGE baselines
mkdir pearlm/pathlm/models/kge_rec
mv KGE/mapper_torchkge.py pearlm/pathlm/data_mappers/.
mv KGE/* pearlm/pathlm/models/kge_rec/.

# Move the updated r_map for ml1m and lfm1m to the correct folder
cp -f fix/r_map_ml1m.txt pearlm/data/ml1m/preprocessed/r_map.txt
cp -f fix/r_map_lfm1m.txt pearlm/data/lfm1m/preprocessed/r_map.txt
mv data/* pearlm/data
rmdir data


touch pearlm/pathlm/data_mappers/__init__.py
mv fix/kg_test_ml1m.txt pearlm/data/ml1m/preprocessed/kg_test.txt
mv fix/kg_train_ml1m.txt pearlm/data/ml1m/preprocessed/kg_train.txt

mv fix/kg_test_lfm1m.txt pearlm/data/lfm1m/preprocessed/kg_test.txt
mv fix/kg_train_lfm1m.txt pearlm/data/lfm1m/preprocessed/kg_train.txt

# Fix for UKGCLM 
mv UKGCLM pearlm/pathlm/models/lm/.
mv fix/main_LP.py pearlm/pathlm/sampling/
mv fix/samplerLP.py pearlm/pathlm/sampling/samplers/

mv data/lfm1m/* pearlm/data/lfm1m/
mv data/ml1m/* pearlm/data/ml1m/
