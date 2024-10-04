import argparse

from kgglm.knowledge_graphs.kg_macros import PRODUCT, USER


def none_or_str(value):
    if value == 'None':
        return None
    return value


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def parse_sampler_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1m',
                        help='One of {ml1m, lfm1m}')
    parser.add_argument('--root_dir', type=str, default='./',
                        help='Working directory to use to compute the datasets')
    parser.add_argument('--data_dirname', type=str, default='data',
                        help='Directory name to use to store the datasets')
    parser.add_argument('--max_n_paths', type=int, default=100,
                        help='Max number of paths sampled for each user.')
    parser.add_argument('--max_hop', type=none_or_int,
                        default=3, help='Max number of hops.')
    parser.add_argument("--itemset_type", type=str, default='inner',
                        help="Choose whether final entity of a path is a product\nin the train interaction set of a user, outer set, or any reachable item {inner,outer,all} respectively")
    parser.add_argument("--collaborative", type=bool, default=False,
                        help="Wether paths should be sampled considering users as intermediate entities")
    parser.add_argument("--with_type", type=bool,
                        default=False, help="Typified paths")
    parser.add_argument('--nproc', type=int, default=4,
                        help='Number of processes to sample in parallel')
    parser.add_argument("--start_type", type=none_or_str,
                        default=USER, help="Start paths with chosen type")
    parser.add_argument("--end_type", type=none_or_str,
                        default=PRODUCT, help="End paths with chosen type")
    # Sample paths for recommendation or Link Prediction?
    parser.add_argument("--task", type=str,
                        default='rec', help="Sample paths for Recommendation or for Knowledge Completion (Link prediction)")
    args = parser.parse_args()
    return args