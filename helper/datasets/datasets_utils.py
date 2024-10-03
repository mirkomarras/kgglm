import csv
import glob
import os
import pickle
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from helper.utils import get_dataset_id2eid, get_model_dir

def get_user_negatives(dataset_name: str) -> Dict[int, List[int]]:
    """
    Returns a dictionary with the user negatives in the dataset, this means the items not interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.

    Args:
        dataset_name (str): 

    Returns:
        Dict[int, List[int]]: Entities ids not interacted with by the user
    """    
    pid2eid = get_dataset_id2eid(dataset_name, what='product')
    ikg_ids = set([int(eid) for eid in set(pid2eid.values())]) # All the ids of products in the kg
    uid_negatives = {}
    # Generate paths for the test set
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        uid_negatives[uid] = [int(pid) for pid in list(set(ikg_ids - set(train_set[uid]) - set(valid_set[uid])))]
    return uid_negatives

def get_user_positives(dataset_name: str) -> Dict[int, List[int]]:
    """
    Returns a dictionary with the user positives in the dataset, this means the items interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.

    Args:
        dataset_name (str): 

    Returns:
        Dict[int, List[int]]: Return user positives for each userid
    """    
    uid_positives = {}
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user positives", colour="green"):
        uid_positives[uid] = list(set(train_set[uid]).union(set(valid_set[uid])))
    return uid_positives

def get_set(dataset_name: str, set_str: str = 'test') -> Dict[int, List[int]]:
    """
    Returns a dictionary containing the user interactions in the selected set {train, valid, test}.
    Note that the ids are the entity ids to be in the same space of the models.

    Args:
        dataset_name (str): 
        set_str (str, optional): which split?. Defaults to 'test'.

    Returns:
        Dict[int, List[int]]: Return item ids for each userid
    """
    data_dir = f"data/{dataset_name}"
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    uid2eid = get_dataset_id2eid(dataset_name, what='user')
    pid2eid = get_dataset_id2eid(dataset_name, what='product')

    # Generate paths for the test set
    curr_set = defaultdict(list)
    with open(f"{data_dir}/preprocessed/{set_str}.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            # user_id starts from 1 in the augmented graph starts from 0
            user_id = int(uid2eid[user_id])
            item_id = int(pid2eid[item_id])  # Converting dataset id to eid
            curr_set[user_id].append(item_id)
    f.close()
    return curr_set

"""{Utils for CAFE, UCPR, PGPR}"""

def save_dataset(dataset, dataset_obj, TMP_DIR):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    if not os.path.exists(TMP_DIR[dataset]):
        os.makedirs(TMP_DIR[dataset])
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset, TMP_DIR):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, LABELS_DIR,mode='train'):
    if mode == 'train':
        label_file = LABELS_DIR[dataset][0]
    elif mode == 'valid':
        label_file = LABELS_DIR[dataset][1]
    elif mode == 'test':
        label_file = LABELS_DIR[dataset][2]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)
    f.close()


def save_kg(dataset, kg, TMP_DIR):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(kg_file)}".')

def load_kg(dataset, TMP_DIR):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    # CHANGED
    kg = pickle.load(open(kg_file,'rb'))
    return kg

def load_embed(dataset_name: str, model_name:str, embed_name: str = 'TransE'):
    if not os.path.exists(os.path.join(get_model_dir(model_name, 'rl'), 'embeddings')):
        # Except for file not found, raise error
        raise FileNotFoundError(
            f'Embedding folder not found. Please run preprocess_embeddings.py first')

    checkpoint = glob.glob(os.path.join(get_model_dir(
        model_name, 'rl'), 'embeddings', f'{embed_name}_structured_{dataset_name}.pkl'))
    assert len(
        checkpoint) != 0, f"[Error]: Please first format previously trained {embed_name} embeddings with preprocess_embeddings.py"
    ckpt = max(checkpoint, key=os.path.getmtime)  # get latest ckpt
    embed = pickle.load(open(ckpt, 'rb'))
    return embed
