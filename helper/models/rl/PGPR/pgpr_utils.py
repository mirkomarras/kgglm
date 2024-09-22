
import csv
import glob
import logging
import logging.handlers
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np
import torch

from helper.utils import get_model_dir

# Dataset names.
ML1M = 'ml1m'
LFM1M = 'lfm1m'
CELL = 'cellphones'
COCO = 'coco'

MODEL = 'pgpr'
TRANSE='TransE'
ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

# Dataset directories.

MODEL_DATASET_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}'
}

# Dataset directories.
DATASET_INFO_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/mapping',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/mapping'
}


VALID_METRICS_FILE_NAME = 'valid_metrics.json'

OPTIM_HPARAMS_METRIC = 'valid_reward'
OPTIM_HPARAMS_LAST_K = 100 # last 100 episodes
LOG_DIR = f'{ROOT_DIR}/results'


LOG_DATASET_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/{MODEL}',
    LFM1M: f'{LOG_DIR}/{LFM1M}/{MODEL}'
}

# for compatibility, CFG_DIR, BEST_CFG_DIR have been modified s,t, they are independent from the dataset
CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/hparams_cfg'
}
BEST_CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/best_hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/best_hparams_cfg'
}
TEST_METRICS_FILE_NAME = 'test_metrics.json'
RECOM_METRICS_FILE_NAME = 'recommender_metrics.json'

RECOM_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{RECOM_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{RECOM_METRICS_FILE_NAME}'
}

TEST_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}'
}
BEST_TEST_METRICS_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}'
}


CONFIG_FILE_NAME = 'config.json'
CFG_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}'
}
BEST_CFG_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}'
}

TRANSE_HPARAMS_FILE = f'transe_{MODEL}_hparams_file.json'
HPARAMS_FILE = f'{MODEL}_hparams_file.json'

# Model result directories.
TMP_DIR = {
    ML1M: f'{MODEL_DATASET_DIR[ML1M]}/tmp',
    LFM1M: f'{MODEL_DATASET_DIR[LFM1M]}/tmp'
}

# Label files.
LABELS = {
    ML1M: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/valid_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl'),
    LFM1M: (TMP_DIR[LFM1M] + '/train_label.pkl', TMP_DIR[LFM1M] + '/valid_label.pkl', TMP_DIR[LFM1M] + '/test_label.pkl')
}

def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_dataset(dataset, dataset_obj):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    if not os.path.exists(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)

def load_dataset(dataset):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj

def save_labels(dataset, labels, mode='train'):
    if not os.path.exists(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'valid':
        label_file = LABELS[dataset][1]
    elif mode == 'test':
        label_file = LABELS[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)
    f.close()

def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'valid':
        label_file = LABELS[dataset][1]
    elif mode == 'test':
        label_file = LABELS[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products

# Receive paths in form (score, prob, [path]) return the last relationship
def get_path_pattern(path):
    return path[-1][-1][0]

def get_pid_to_kgid_mapping(dataset_name):
    if dataset_name == "ml1m":
        file = open(MODEL_DATASET_DIR[dataset_name] + "/entities/mappings/movie.txt", "r")
    elif dataset_name == "lfm1m":
        file = open(MODEL_DATASET_DIR[dataset_name] + "/entities/mappings/song.txt", "r")
    else:
        print("Dataset mapping not found!")
        exit(-1)
    reader = csv.reader(file, delimiter=' ')
    dataset_pid2kg_pid = {}
    next(reader, None)
    for row in reader:
        if dataset_name == "ml1m" or dataset_name == "lfm1m":
            dataset_pid2kg_pid[int(row[0])] = int(row[1])
    file.close()
    return dataset_pid2kg_pid

def get_validation_pids(dataset_name):
    if not os.path.isfile(os.path.join(MODEL_DATASET_DIR[dataset_name], 'valid.txt')):
        return []
    validation_pids = defaultdict(set)
    with open(os.path.join(MODEL_DATASET_DIR[dataset_name], 'valid.txt')) as valid_file:
        reader = csv.reader(valid_file, delimiter=" ")
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            validation_pids[uid].add(pid)
    valid_file.close()
    return validation_pids

def get_uid_to_kgid_mapping(dataset_name):
    dataset_uid2kg_uid = {}
    with open(MODEL_DATASET_DIR[dataset_name] + "/entities/mappings/user.txt", 'r') as file:
        reader = csv.reader(file, delimiter=" ")
        next(reader, None)
        for row in reader:
            if dataset_name == "ml1m" or dataset_name == "lfm1m":
                uid_review = int(row[0])
            uid_kg = int(row[1])
            dataset_uid2kg_uid[uid_review] = uid_kg
    return dataset_uid2kg_uid

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    if not os.path.exists(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    pickle.dump(kg, open(kg_file, 'wb'))

def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    # CHANGED
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def shuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)


def load_embed(dataset_name: str, embed_name: str = None):
    if not os.path.exists(os.path.join(get_model_dir('PGPR', 'rl'), 'embeddings')):
        # Except for file not found, raise error
        raise FileNotFoundError(f'Embedding folder not found. Please run preprocess_embeddings.py first')
    
    checkpoint = glob.glob(os.path.join(get_model_dir('PGPR', 'rl'), 'embeddings', f'{embed_name}_structured_{dataset_name}.pkl'))
    assert len(checkpoint) != 0, f"[Error]: Please first format previously trained {embed_name} embeddings with preprocess_embeddings.py"
    ckpt = max(checkpoint, key=os.path.getmtime)  # get latest ckpt
    embed = pickle.load(open(ckpt, 'rb'))
    return embed
