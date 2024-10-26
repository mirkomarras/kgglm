import gzip
import os
import pickle

import numpy as np

from kgglm.data.knowledge_graph.kg_macros import INTERACTION, ROOT_DIR

CAFE = "cafe"
MODEL = CAFE
# Dataset directories.
DATA_DIR = {
    dset: f"{ROOT_DIR}/data/{dset}/preprocessed/{MODEL}" for dset in INTERACTION
}

OPTIM_HPARAMS_METRIC = "avg_valid_loss"
VALID_METRICS_FILE_NAME = "valid_metrics.json"

LOG_DIR = f"{ROOT_DIR}/results"

LOG_DATASET_DIR = {dset: f"{LOG_DIR}/{dset}/{MODEL}" for dset in INTERACTION}

# for compatibility, CFG_DIR, BEST_CFG_DIR have been modified s,t, they are independent from the dataset
CFG_DIR = {dset: f"{LOG_DATASET_DIR[dset]}/hparams_cfg" for dset in INTERACTION}
BEST_CFG_DIR = {
    dset: f"{LOG_DATASET_DIR[dset]}/best_hparams_cfg" for dset in INTERACTION
}
TEST_METRICS_FILE_NAME = "test_metrics.json"
RECOM_METRICS_FILE_NAME = "recommender_metrics.json"

RECOM_METRICS_FILE_PATH = {
    dset: f"{CFG_DIR[dset]}/{RECOM_METRICS_FILE_NAME}" for dset in INTERACTION
}

TEST_METRICS_FILE_PATH = {
    dset: f"{CFG_DIR[dset]}/{TEST_METRICS_FILE_NAME}" for dset in INTERACTION
}
BEST_TEST_METRICS_FILE_PATH = {
    dset: f"{BEST_CFG_DIR[dset]}/{TEST_METRICS_FILE_NAME}" for dset in INTERACTION
}


CONFIG_FILE_NAME = "config.json"
CFG_FILE_PATH = {dset: f"{CFG_DIR[dset]}/{CONFIG_FILE_NAME}" for dset in INTERACTION}
BEST_CFG_FILE_PATH = {
    dset: f"{BEST_CFG_DIR[dset]}/{CONFIG_FILE_NAME}" for dset in INTERACTION
}

HPARAMS_FILE = f"{MODEL}_hparams_file.json"

# Model result directories.
TMP_DIR = {dset: f"{DATA_DIR[dset]}/tmp" for dset in INTERACTION}

LABEL_FILE = {
    dset: (
        DATA_DIR[dset] + "/train.txt.gz",
        DATA_DIR[dset] + "/valid.txt.gz",
        DATA_DIR[dset] + "/test.txt.gz",
    )
    for dset in INTERACTION
}


def save_embed(dataset, embed):
    if not os.path.isdir(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    embed_file = TMP_DIR[dataset] + "/embed.pkl"
    pickle.dump(embed, open(embed_file, "wb"))
    print(f'File is saved to "{os.path.abspath(embed_file)}".')


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + "/kg.pkl"
    kg = pickle.load(open(kg_file, "rb"))
    return kg


def load_user_products(dataset, up_type="pos"):
    up_file = "{}/user_products_{}.npy".format(TMP_DIR[dataset], up_type)
    with open(up_file, "rb") as f:
        up = np.load(f)
    return up


def save_user_products(dataset, up, up_type="pos"):
    up_file = "{}/user_products_{}.npy".format(TMP_DIR[dataset], up_type)
    with open(up_file, "wb") as f:
        np.save(f, up)
    print(f'File is saved to "{os.path.abspath(up_file)}".')


def load_labels(dataset, mode="train"):
    if mode == "train":
        label_file = LABEL_FILE[dataset][0]
    elif mode == "valid":
        label_file = LABEL_FILE[dataset][1]
    elif mode == "test":
        label_file = LABEL_FILE[dataset][2]
    else:
        raise Exception("mode should be one of {train, test}.")
    # user_products = pickle.load(open(label_file, 'rb'))
    labels = {}  # key: user_id, value: list of item IDs.
    with gzip.open(label_file, "rb") as f:
        for line in f:
            cells = line.decode().strip().split("\t")
            labels[int(cells[0])] = [int(x) for x in cells[1:]]
    return labels


def load_path_count(dataset):
    count_file = TMP_DIR[dataset] + "/path_count.pkl"
    count = pickle.load(open(count_file, "rb"))
    return count


def save_path_count(dataset, count):
    count_file = TMP_DIR[dataset] + "/path_count.pkl"
    pickle.dump(count, open(count_file, "wb"))
