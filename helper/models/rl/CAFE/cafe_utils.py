import os
import pickle
import numpy as np
import gzip

from helper.knowledge_graphs.kg_macros import ML1M, LFM1M

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

CAFE='cafe'
MODEL=CAFE
# Dataset directories.
DATA_DIR = {
    ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
    LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}'
}

OPTIM_HPARAMS_METRIC = 'avg_valid_loss'
VALID_METRICS_FILE_NAME = 'valid_metrics.json'


LOG_DIR = f'{ROOT_DIR}/results'


LOG_DATASET_DIR = {
    ML1M: f'{LOG_DIR}/{ML1M}/{MODEL}',
    LFM1M: f'{LOG_DIR}/{LFM1M}/{MODEL}'
}

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

HPARAMS_FILE = f'{MODEL}_hparams_file.json'

# Model result directories.
TMP_DIR = {
    ML1M: f'{DATA_DIR[ML1M]}/tmp',
    LFM1M: f'{DATA_DIR[LFM1M]}/tmp',
}

LABEL_FILE = {
    ML1M: (DATA_DIR[ML1M] + '/train.txt.gz', DATA_DIR[ML1M] + '/valid.txt.gz', DATA_DIR[ML1M] + '/test.txt.gz'),
    LFM1M: (DATA_DIR[LFM1M] + '/train.txt.gz', DATA_DIR[LFM1M] + '/valid.txt.gz', DATA_DIR[LFM1M] + '/test.txt.gz')
}

def save_embed(dataset, embed):
    if not os.path.isdir(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    embed_file = TMP_DIR[dataset] + '/embed.pkl'
    pickle.dump(embed, open(embed_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(embed_file)}".')

def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(kg_file)}".')

def load_user_products(dataset, up_type='pos'):
    up_file = '{}/user_products_{}.npy'.format(TMP_DIR[dataset], up_type)
    with open(up_file, 'rb') as f:
        up = np.load(f)
    return up

def save_user_products(dataset, up, up_type='pos'):
    up_file = '{}/user_products_{}.npy'.format(TMP_DIR[dataset], up_type)
    with open(up_file, 'wb') as f:
        np.save(f, up)
    print(f'File is saved to "{os.path.abspath(up_file)}".')

def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABEL_FILE[dataset][0]
    elif mode == 'valid':
        label_file = LABEL_FILE[dataset][1]
    elif mode == 'test':
        label_file = LABEL_FILE[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    # user_products = pickle.load(open(label_file, 'rb'))
    labels = {}  # key: user_id, value: list of item IDs.
    with gzip.open(label_file, 'rb') as f:
        for line in f:
            cells = line.decode().strip().split('\t')
            labels[int(cells[0])] = [int(x) for x in cells[1:]]
    return labels

def load_path_count(dataset):
    count_file = TMP_DIR[dataset] + '/path_count.pkl'
    count = pickle.load(open(count_file, 'rb'))
    return count

def save_path_count(dataset, count):
    count_file = TMP_DIR[dataset] + '/path_count.pkl'
    pickle.dump(count, open(count_file, 'wb'))

def makedirs(dataset_name):
    os.makedirs(BEST_CFG_DIR[dataset_name], exist_ok=True)
    os.makedirs(CFG_DIR[dataset_name], exist_ok=True)
