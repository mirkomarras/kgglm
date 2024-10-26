import os
import pickle

from kgglm.data.knowledge_graph.kg_macros import INTERACTION, ROOT_DIR

MODEL = "pgpr"

# Dataset directories.

MODEL_DATASET_DIR = {
    dset: f"{ROOT_DIR}/data/{dset}/preprocessed/{MODEL}" for dset in INTERACTION
}

# Dataset directories.
DATASET_INFO_DIR = {
    dset: f"{ROOT_DIR}/data/{dset}/preprocessed/mapping" for dset in INTERACTION
}


VALID_METRICS_FILE_NAME = "valid_metrics.json"

OPTIM_HPARAMS_METRIC = "valid_reward"
OPTIM_HPARAMS_LAST_K = 100  # last 100 episodes
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

TRANSE_HPARAMS_FILE = f"transe_{MODEL}_hparams_file.json"
HPARAMS_FILE = f"{MODEL}_hparams_file.json"

# Model result directories.
TMP_DIR = {dset: f"{MODEL_DATASET_DIR[dset]}/tmp" for dset in INTERACTION}

LABEL_FILE = {
    dset: (
        MODEL_DATASET_DIR[dset] + "/train.txt.gz",
        MODEL_DATASET_DIR[dset] + "/valid.txt.gz",
        MODEL_DATASET_DIR[dset] + "/test.txt.gz",
    )
    for dset in INTERACTION
}


def load_labels(dataset, mode="train"):
    if mode == "train":
        label_file = LABEL_FILE[dataset][0]
    elif mode == "valid":
        label_file = LABEL_FILE[dataset][1]
    elif mode == "test":
        label_file = LABEL_FILE[dataset][2]
    else:
        raise Exception("mode should be one of {train, test}.")
    user_products = pickle.load(open(label_file, "rb"))
    return user_products


# Receive paths in form (score, prob, [path]) return the last relationship
def get_path_pattern(path):
    return path[-1][-1][0]


def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)
