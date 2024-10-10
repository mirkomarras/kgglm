DATASET=$1
NPATHS=$2
HOPS=$3
NPROC=$4


ROOT_DIR='.'
DATA_DIR='data'
TASK='finetuneREC'

LOGDIR=$(echo "datasetREC_${DATASET}__hops_${HOPS}__npaths_${NPATHS}")

pip install . && python3 kgglm/sampling/main.py --root_dir $ROOT_DIR --data_dir $DATA_DIR --log_dir $LOGDIR --dataset $DATASET --max_n_paths $NPATHS --max_hop $HOPS --collaborative TRUE --nproc $NPROC --start_type user --end_type product --itemset_type inner --task rec
find $ROOT_DIR/$DATA_DIR/sampled/$DATASET/$LOGDIR -name '*.txt' -exec cat {} \; >> concatenated_rw_file.txt
mkdir -p $ROOT_DIR/$DATA_DIR/$DATASET/paths_random_walk
mv concatenated_rw_file.txt $ROOT_DIR/$DATA_DIR/$DATASET/paths_random_walk/paths_${TASK}_${NPATHS}_${HOPS}.txt
