import argparse
import os

from helper.datasets.KARSDataset import KARSDataset
from helper.evaluation.eval_utils import get_set
from helper.knowledge_graphs.pgpr_kg import PGPRKnowledgeGraph
from helper.models.rl.PGPR.pgpr_utils import (TMP_DIR, load_dataset,
                                              save_dataset, save_kg,
                                              save_labels)


def generate_labels(dataset: str, mode: str='train') -> None:
    #USE GET_SET() FUNCTION AND CHEKC THAT IDS ARE CORRECT IN THE PIPE
    user_products = get_set(dataset, mode)
    user_products = {int(k): [int(v) for v in user_products[k]] for k in user_products}
    save_labels(dataset, user_products, mode=mode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ml1m", help='ML1M')
    args = parser.parse_args()

    # Create AmazonDataset instance for dataset.
    # ========== BEGIN ========== #
    print('Load', args.dataset, 'dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = KARSDataset(args.dataset)
    save_dataset(args.dataset, dataset)
    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print('Create', args.dataset, 'knowledge graph from dataset...')
    dataset = load_dataset(args.dataset)
    kg = PGPRKnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)
    # =========== END =========== #

    # Generate train/test labels.
    # ========== BEGIN ========== #
    print('Generate', args.dataset, 'train/test labels.')
    generate_labels(args.dataset, 'train')
    generate_labels(args.dataset, 'valid')
    generate_labels(args.dataset, 'test')
    # =========== END =========== #


if __name__ == '__main__':
    main()
