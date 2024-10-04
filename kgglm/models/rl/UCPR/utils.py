import os
import pickle

from kgglm.knowledge_graphs.kg_macros import INTERACTION, ROOT_DIR


MODEL = 'ucpr'

# Dataset directories.
DATASET_DIR = {
    dset: f'{ROOT_DIR}/data/{dset}/preprocessed/{MODEL}' for dset in INTERACTION
}

LOG_DIR = f'{ROOT_DIR}/results'

LOG_DATASET_DIR = {
    dset: f'{LOG_DIR}/{dset}/{MODEL}' for dset in INTERACTION
}

VALID_METRICS_FILE_NAME = 'valid_metrics.json'

OPTIM_HPARAMS_METRIC = 'valid_reward'
OPTIM_HPARAMS_LAST_K = 100 # last 100 episodes


# for compatibility, CFG_DIR, BEST_CFG_DIR have been modified s,t, they are independent from the dataset
CFG_DIR = {
    dset: f'{LOG_DATASET_DIR[dset]}/hparams_cfg' for dset in INTERACTION
}
BEST_CFG_DIR = {
    dset: f'{LOG_DATASET_DIR[dset]}/best_hparams_cfg' for dset in INTERACTION
}
TEST_METRICS_FILE_NAME = 'test_metrics.json'
RECOM_METRICS_FILE_NAME = 'recommender_metrics.json'

RECOM_METRICS_FILE_PATH = {
    dset: f'{CFG_DIR[dset]}/{RECOM_METRICS_FILE_NAME}' for dset in INTERACTION
}

TEST_METRICS_FILE_PATH = {
    dset: f'{CFG_DIR[dset]}/{TEST_METRICS_FILE_NAME}' for dset in INTERACTION
}
BEST_TEST_METRICS_FILE_PATH = {
    dset: f'{BEST_CFG_DIR[dset]}/{TEST_METRICS_FILE_NAME}' for dset in INTERACTION
}


CONFIG_FILE_NAME = 'config.json'
CFG_FILE_PATH = {
    dset: f'{CFG_DIR[dset]}/{CONFIG_FILE_NAME}' for dset in INTERACTION
}
BEST_CFG_FILE_PATH = {
    dset: f'{BEST_CFG_DIR[dset]}/{CONFIG_FILE_NAME}' for dset in INTERACTION
}

HPARAMS_FILE = f'{MODEL}_hparams_file.json'

# Model result directories.
TMP_DIR = {
    dset: f'{DATA_DIR[dset]}/tmp' for dset in INTERACTION
}

SAVE_MODEL_DIR = {
    dset: f'{LOG_DATASET_DIR[dset]}/save' for dset in INTERACTION
}
EVALUATION = {
    dset: f'{LOG_DATASET_DIR[dset]}/eva_pre' for dset in INTERACTION
}
EVALUATION_2 = {
    dset: f'{LOG_DATASET_DIR[dset]}/eval' for dset in INTERACTION
}
CASE_ST = {
    dset: f'{LOG_DATASET_DIR[dset]}/case_st' for dset in INTERACTION
}
TEST = {
    dset: f'{LOG_DATASET_DIR[dset]}/test' for dset in INTERACTION
}

# Label files.
LABELS = {
    ML1M: (TMP_DIR[dset] + '/train_label.pkl', TMP_DIR[dset] + '/valid_label.pkl', TMP_DIR[dset] + '/test_label.pkl') for dset in INTERACTION
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
