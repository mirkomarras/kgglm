from __future__ import absolute_import, division, print_function
import logging
import logging.handlers
import os
import pickle
import sys

from helper.knowledge_graphs.kg_macros import ML1M, LFM1M, ROOT_DIR, DATASETS


MODEL = 'ucpr'

# Dataset directories.
DATASET_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}',
}

# Model result directories.
TMP_DIR = {
    ML1M: f'{DATASET_DIR[ML1M]}/tmp',
    LFM1M: f'{DATASET_DIR[LFM1M]}/tmp',
}

LOG_DIR = f'{ROOT_DIR}/results'

LOG_DATASET_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/{MODEL}',
    LFM1M: f'{LOG_DIR}/{LFM1M}/{MODEL}',
}

VALID_METRICS_FILE_NAME = 'valid_metrics.json'

OPTIM_HPARAMS_METRIC = 'valid_reward'
OPTIM_HPARAMS_LAST_K = 100 # last 100 episodes


# for compatibility, CFG_DIR, BEST_CFG_DIR have been modified s,t, they are independent from the dataset
CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/hparams_cfg',
}
BEST_CFG_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/best_hparams_cfg',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/best_hparams_cfg',

}

TEST_METRICS_FILE_NAME = 'test_metrics.json'
RECOM_METRICS_FILE_NAME = 'recommender_metrics.json'

RECOM_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{RECOM_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{RECOM_METRICS_FILE_NAME}',
}

TEST_METRICS_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}',

}
BEST_TEST_METRICS_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{TEST_METRICS_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{TEST_METRICS_FILE_NAME}',

}
CONFIG_FILE_NAME = 'config.json'
CFG_FILE_PATH = {
    ML1M: f'{CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}',

}
BEST_CFG_FILE_PATH = {
    ML1M: f'{BEST_CFG_DIR[ML1M]}/{CONFIG_FILE_NAME}',
    LFM1M: f'{BEST_CFG_DIR[LFM1M]}/{CONFIG_FILE_NAME}',

}

HPARAMS_FILE = f'{MODEL}_hparams_file.json'
SAVE_MODEL_DIR = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/save',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/save',

}
EVALUATION = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/eva_pre',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/eva_pre',

}
EVALUATION_2 = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/eval',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/eval',

}
CASE_ST = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/case_st',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/case_st',

}

TEST = {
    ML1M: f'{LOG_DATASET_DIR[ML1M]}/test',
    LFM1M: f'{LOG_DATASET_DIR[LFM1M]}/test',

}
# Label files.
LABELS = {
    ML1M: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/valid_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl'),
    LFM1M: (TMP_DIR[LFM1M] + '/train_label.pkl', TMP_DIR[LFM1M] + '/valid_label.pkl', TMP_DIR[LFM1M] + '/test_label.pkl'),
}

# UCPR SPECIFIC RELATIONS
PADDING = 'padding'



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


def ensure_dataset_name(dataset_name):
    if dataset_name not in DATASETS:
        print("Dataset not recognised, check for typos")
        exit(-1)
    return

def get_model_data_dir(model_name, dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/preprocessed/{model_name}/"

def get_data_dir(dataset_name):
    ensure_dataset_name(dataset_name)
    return f"data/{dataset_name}/ ed/"
