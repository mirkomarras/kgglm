import csv
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
from helper.utils import get_dataset_id2eid


def get_user_negatives(dataset_name: str) -> Dict[int, List[int]]:
    """
    Returns a dictionary with the user negatives in the dataset, this means the items not interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.
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


def get_set(dataset_name: str, set_str: str = 'test') -> Dict[int, List[int]]:
    """
    Returns a dictionary containing the user interactions in the selected set {train, valid, test}.
    Note that the ids are the entity ids to be in the same space of the models.
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
            user_id = int(uid2eid[user_id])  # user_id starts from 1 in the augmented graph starts from 0
            item_id = int(pid2eid[item_id])  # Converting dataset id to eid
            curr_set[user_id].append(item_id)
    f.close()
    return curr_set


def get_user_positives(dataset_name: str) -> Dict[int, List[int]]:
    """
    Returns a dictionary with the user positives in the dataset, this means the items interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.
    """
    uid_positives = {}
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        uid_positives[uid] = list(set(train_set[uid]).union(set(valid_set[uid])))
    return uid_positives