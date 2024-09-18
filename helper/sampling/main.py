import os

from transformers import set_seed

from helper.sampling.parser import parse_sampler_args
from helper.sampling.samplers.sampler import KGsampler
from helper.sampling.samplers.samplerLP import KGSamplerLinkPrediction
from helper.utils import SEED, get_data_dir, get_dataset_info_dir

if __name__ == '__main__':
    set_seed(SEED)
    args=parse_sampler_args()

    # root dir is current directory (according to the location from where this script is run)
    # e.g. if helper/sampling/main.py then ./ translates to helper
    ROOT_DIR = args.root_dir
    ROOT_DATA_DIR = os.path.join(ROOT_DIR, args.data_dirname)
    SAVE_DIR = os.path.join(ROOT_DATA_DIR, 'sampled')
    MAX_HOP = args.max_hop
    N_PATHS = args.max_n_paths
    itemset_type = args.itemset_type
    COLLABORATIVE = args.collaborative
    NPROC = args.nproc
    WITH_TYPE = args.with_type

    # Dataset directories
    dataset_name = args.dataset
    dirpath = get_data_dir(dataset_name)
    data_dir_mapping = get_dataset_info_dir(dataset_name)

    # Sample paths according to the task
    if args.task=='rec':
        kg = KGsampler(args.dataset, save_dir=SAVE_DIR, data_dir=data_dir_mapping)
        LOGDIR = f'dataset_{args.dataset}__hops_{MAX_HOP}__npaths_{N_PATHS}'
    else: 
        kg = KGSamplerLinkPrediction(args.dataset, save_dir=SAVE_DIR, data_dir=data_dir_mapping)
        LOGDIR = f'datasetLP_{args.dataset}__hops_{MAX_HOP}__npaths_{N_PATHS}'
    
    print('Closed destination item set: ', itemset_type)
    print('Collaborative filtering: ', args.collaborative)

    kg.random_walk_sampler(max_hop=MAX_HOP, 
                           logdir=LOGDIR,
                           ignore_rels=set(), 
                           max_paths=N_PATHS, 
                           itemset_type=itemset_type,
                           collaborative=COLLABORATIVE,
                           nproc=NPROC,
                           with_type=WITH_TYPE,
                           start_ent_type=args.start_type,
                           end_ent_type=args.end_type)
