DATASET=$1
NPATHS=$2
HOPS=$3
NPROC=$4


ROOT_DIR='./'
DATA_DIR='data'
TASK='finetuneLP'

LOGDIR=$(echo "datasetLP_${DATASET}__hops_${HOPS}__npaths_${NPATHS}")

pip install . && python3 helper/sampling/main_LP.py --root_dir $ROOT_DIR --data_dir $DATA_DIR --dataset $DATASET --max_n_paths $NPATHS --max_hop $HOPS --nproc $NPROC --start_type product --end_type entity --itemset_type all --task lp
find $ROOT_DIR/$DATA_DIR/sampledLP/$DATASET/$LOGDIR -name '*.txt' -exec cat {} \; >> concatenated_rw_file.txt
python3 helper/sampling/prune_dataset.py --filepath concatenated_rw_file.txt
mkdir -p $ROOT_DIR/$DATA_DIR/$DATASET/paths_random_walk
mv concatenated_rw_file_pruned.txt $ROOT_DIR/$DATA_DIR/$DATASET/paths_random_walk/paths_${TASK}_${NPATHS}_${HOPS}.txt
rm concatenated_rw_file.txt