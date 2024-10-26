from __future__ import absolute_import, division, print_function

import argparse
import gzip
import os

from kgglm.data.dataset.dataloader_ucpr import Dataset
from kgglm.data.dataset.datasets_utils import (
    load_dataset,
    save_dataset,
    save_kg,
    save_labels,
)
from kgglm.data.knowledge_graph.kg_ucpr import KnowledgeGraph
from kgglm.model.rl.UCPR.utils import DATASET_DIR, TMP_DIR


def generate_labels(dataset, mode="train"):
    review_file = f"{DATASET_DIR[dataset]}/{mode}.txt.gz"
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(review_file, "r") as f:
        for line in f:
            line = line.decode("utf-8").strip()
            arr = line.split("\t")
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(dataset, user_products, mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lfm1m", help="ML1M")
    args = parser.parse_args()

    # ========== BEGIN ========== #
    print("Load", args.dataset, "dataset from file...")
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = Dataset(args)
    save_dataset(args.dataset, dataset)
    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print("Create", args.dataset, "knowledge graph from dataset...")
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)
    # =========== END =========== #

    # Generate train/test labels.
    # ========== BEGIN ========== #
    print("Generate", args.dataset, "train/test labels.")
    generate_labels(args.dataset, "train")
    generate_labels(args.dataset, "valid")
    generate_labels(args.dataset, "test")
    # =========== END =========== #


if __name__ == "__main__":
    main()
