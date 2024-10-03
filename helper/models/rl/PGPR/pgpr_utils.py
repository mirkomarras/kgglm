import os
import pickle

from helper.knowledge_graphs.kg_macros import LFM1M, ML1M, ROOT_DIR

MODEL = 'pgpr'

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

def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)



