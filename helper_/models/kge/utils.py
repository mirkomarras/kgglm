import pandas as pd
import os
import numpy as np
from helper.data_mappers.mapper_torchkge import dataset_preprocessing
from torchkge import KnowledgeGraph
import torch
from helper.evaluation.eval_metrics import ndcg_at_k,mmr_at_k
from collections import defaultdict
from typing import List, Tuple, Dict
from helper.utils import get_dataset_id2eid
from tqdm import tqdm



def get_weight_dir(method_name: str, dataset: str):
    weight_dir = os.path.join("helper/models/kge/",method_name,"weight_dir")
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)
    return weight_dir

def get_weight_ckpt_dir(method_name: str,dataset: str):
    weight_dir_ckpt = os.path.join("helper/models/kge/",method_name,"weight_dir_ckpt")
    if not os.path.isdir(weight_dir_ckpt):
        os.makedirs(weight_dir_ckpt)
    return weight_dir_ckpt

def get_test_uids(dataset):
    preprocessed_torchkge=f"data/{dataset}/preprocessed/torchkge"
    test_path = f"{preprocessed_torchkge}/triplets_test.txt"
    kg_df_test = pd.read_csv(test_path, sep="\t")
    kg_df_test.rename(columns={"0":"from","1":"to","2":"rel"},inplace=True)
    uids=np.unique(kg_df_test['from'])
    return uids

def get_log_dir(method_name: str):
    log_dir = os.path.join("helper/models/kge/",method_name,"log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir

def load_kg(dataset): 
    preprocessed_torchkge=f"data/{dataset}/preprocessed/torchkge"
    dataset_preprocessing(dataset)
    e_df_withUsers=pd.read_csv(f"{preprocessed_torchkge}/e_map_with_users.txt", sep="\t")
    kg_df = pd.read_csv(f"{preprocessed_torchkge}/kg_final_updated.txt", sep="\t")
    kg_df.rename(columns={"entity_head":"from","entity_tail":"to","relation":"rel"},inplace=True)
    kg_train=KnowledgeGraph(df=kg_df,ent2ix=dict(zip(e_df_withUsers['eid'],e_df_withUsers['eid'])))
    return kg_train

def get_preprocessed_torchkge_path(dataset):
    preprocessed_torchkge=f"data/{dataset}/preprocessed/torchkge"
    return preprocessed_torchkge

def get_users_positives(path): 
    users_positives=dict()
    with open(f"{path}/triplets_train_valid.txt","r") as f:
        for i,row in enumerate(f):
            if i==0:
                continue
            uid,pid,_=row.split("\t")
            uid=int(uid)
            pid=int(pid)
            if uid in users_positives:
                users_positives[uid].append(pid)
            else:
                users_positives[uid]=[pid]
    return users_positives
 
def remap_topks2datasetid(args,topks):
    """load entities and user_mapping"""
    e_new_df=pd.read_csv(f"{args.preprocessed_torchkge}/e_map_with_users.txt",sep="\t")
    user_mapping=pd.read_csv(f"{args.preprocessed_torchkge}/user_mapping.txt",sep="\t")

    """create the correct mapping"""
    torchkgeid2datasetuid={eid:entity for eid, entity in zip(e_new_df['eid'],e_new_df['entity'])} # mapping uid
    datasetid2useruid={int(datasetid):int(userid) for datasetid, userid in zip(user_mapping['rating_id'],user_mapping['new_id'])}

    """Mapping users to correct uid"""
    topks={int(datasetid2useruid[int(torchkgeid2datasetuid[key])]):values for key, values in topks.items()}
    
    return topks



"""Link prediction Utilities"""

def get_set_lp(dataset,split):
    if dataset == 'ml1m' or dataset == 'lfm1m':
        path = os.path.join('data', dataset, 'preprocessed', f'kg_{split}.txt')
    else:
        path = os.path.join('data', dataset, f'kg_{split}.txt')

    df=pd.read_csv(path,sep="\t")
    test_data=df.to_numpy()
    curr_set=defaultdict(list)
    for triplet in test_data:
        # triplet[0] = e1
        # triplet[1] = r
        # triplet[2] = e2
        curr_set[(triplet[0],triplet[1])].append(triplet[2])
    return curr_set

# def load_kg_lp(dataset,split):
#     if dataset=='ml1m' or dataset=='lfm1m':
#         path=os.path.join('data',dataset,'preprocessed',f'kg_{split}.txt')
#         e_df=pd.read_csv(os.path.join('data',dataset,'preprocessed','e_map.txt'),sep="\t")
#     else:
#         path=os.path.join('data',dataset,f'kg_{split}.txt')
#         e_df = pd.read_csv(os.path.join('data', dataset,'entities.dict'), sep="\t", names=['eid','entity'])

#     kg_df=pd.read_csv(path,sep="\t")
#     kg_df.rename(columns={"entity_head":"from","entity_tail":"to","relation":"rel"},inplace=True)
#     kg=KnowledgeGraph(df=kg_df,ent2ix=dict(zip(e_df['eid'],e_df['eid'])))
#     return kg


def get_users_positives_lp(dataset):
    users_positives=defaultdict(list)
    if dataset=='ml1m' or dataset=='lfm1m':
        path=os.path.join('data',dataset,'preprocessed',f'kg_train.txt') # it contains train + valid as rec task
    else:
        path=os.path.join('data',dataset,f'kg_train.txt')
    with open(path,"r") as f:
        for i,row in enumerate(f):
            if i==0:
                continue
            e1,r,e2 =row.split("\t")
            e1 = int(e1)
            e2 = int(e2)
            r  = int(r)
            if e2 not in users_positives[(e1,r)]:
                users_positives[(e1,r)].append(e2)
    return users_positives

def build_kg_triplets(dataset):
    kg_files=set(['kg_train.txt','kg_test.txt'])
    mapping_files=set(['entities.dict','relations.dict'])
    dir_files=set(os.listdir(os.path.join('data',dataset)))
    if not kg_files.issubset(dir_files) and mapping_files.issubset(dir_files): # if we don't have kg_train and kg_test file but we have relations.dict and entities.dict
        relations_dict=pd.read_csv(os.path.join('data',dataset,'relations.dict'),sep="\t",header=None); relations_dict=relations_dict.to_numpy()
        entities_dict = pd.read_csv(os.path.join('data',dataset,'entities.dict'),sep="\t",header=None); entities_dict=entities_dict.to_numpy()
        relation2rid={rel: rid for rid,rel in relations_dict}
        entity2eid = {entity: eid for eid, entity in entities_dict}

        # Load train.txt, test.txt and valid.txt to do the mapping
        train_txt=pd.read_csv(os.path.join('data',dataset,'train.txt'),sep="\t"); train_txt=train_txt.to_numpy()
        valid_txt = pd.read_csv(os.path.join('data', dataset, 'valid.txt'), sep="\t"); valid_txt=valid_txt.to_numpy()
        test_txt = pd.read_csv(os.path.join('data', dataset, 'test.txt'), sep="\t"); test_txt=test_txt.to_numpy()
        train_txt=np.array([[entity2eid[head], relation2rid[rel], entity2eid[tail]] for head, rel, tail in train_txt])
        valid_txt = np.array([[entity2eid[head], relation2rid[rel], entity2eid[tail]] for head, rel, tail in valid_txt])
        test_txt = np.array([[entity2eid[head], relation2rid[rel], entity2eid[tail]] for head, rel, tail in test_txt])

        # Save the new kg_train, kg_test files ready to be used
        train_txt=np.concatenate([train_txt,valid_txt])
        kg_train=pd.DataFrame(train_txt); kg_train.to_csv(os.path.join('data',dataset,'kg_train.txt'),sep="\t", index=False,header=['entity_head','relation','entity_tail'])
        kg_test = pd.DataFrame(test_txt);kg_test.to_csv(os.path.join('data', dataset, 'kg_test.txt'),sep="\t", index=False,header=['entity_head','relation','entity_tail'])
        return True
    else:
        return False

def mr_at_k(hit_list: List[int], k: int) -> float:
    r = np.asfarray(hit_list)[:k]
    hit_idxs = np.nonzero(r)
    if len(hit_idxs[0]) > 0:
        return hit_idxs[0][0] + 1
    return 0.

def metrics_lp(test_labels, top_k):
    mr=[]
    mrr=[]
    ndcg=[]
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []
    k=10

    for e1_r, topk in top_k.items():
        hits = []
        for e2 in topk[:k]:
            hits.append(1 if e2 in test_labels[e1_r] else 0)

        # If the model has predicted less than 10 items pad with zeros
        while len(hits) < k:
            hits.append(0)

        # Calculate Hits@1, Hits@3, and Hits@10
        hits_at_1.append(1 if sum(hits[:1]) > 0 else 0)
        hits_at_3.append(1 if sum(hits[:3]) > 0 else 0)
        hits_at_10.append(1 if sum(hits[:10]) > 0 else 0)

        mr.append(mr_at_k(hits,k=10))
        mrr.append(mmr_at_k(hits,k=10))
        ndcg.append(ndcg_at_k(hits, k=10))

    mr = np.mean(mr)
    mrr = np.mean(mrr)
    ndcg = np.mean(ndcg)
    hits_at_1_rate = np.mean(hits_at_1)
    hits_at_3_rate = np.mean(hits_at_3)
    hits_at_10_rate = np.mean(hits_at_10)

    results={
        "ndcg":round(ndcg,2),
        "mr":round(mr,2),
        "mrr":round(mrr,2),
        "hits@1":round(hits_at_1_rate*100,2),
        "hits@3":round(hits_at_3_rate*100,2),
        "hits@10":round(hits_at_10_rate*100,2)
    }

    print(f"\n{results}")

    return results

"""Link Prediction on UKGCLM"""
# passed to get_kg_positives_and_tokens_ids_lp but never used in practice
def get_set_entities(dataset):
    if dataset == 'ml1m' or dataset == 'lfm1m':
        path_train = os.path.join('data', dataset, 'preprocessed',
                                  f'kg_train.txt')  # it contains train + valid as rec task
        path_test = os.path.join('data', dataset, 'preprocessed', f'kg_test.txt')
    else:
        path_train = os.path.join('data', dataset, f'kg_train.txt')
        path_test = os.path.join('data', dataset, f'kg_test.txt')

    kg_df_train = pd.read_csv(path_train, sep="\t")
    kg_df_test = pd.read_csv(path_test, sep="\t")

    kg_train = kg_df_train.to_numpy()
    kg_test = kg_df_test.to_numpy()

    all_possible_entities = np.unique(np.concatenate(
        [kg_train[:, 0], kg_train[:, 2], kg_test[:, 0], kg_test[:, 2]]))  # prendo tutte le possibili entità

    return all_possible_entities

def get_kg_positives_and_tokens_ids_lp(dataset_name: str, tokenizer) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Returns a dictionary with the user negatives in the dataset, this means the items not interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.
    And a dictionary with the user negatives tokens ids converted
    """
    # Fetch User Negatives
    kg_train = get_set_lp(dataset_name, 'train') # it contain train and valid
    all_entities=get_set_entities(dataset_name)
    product_entities = [int(h) for h in get_dataset_id2eid(dataset_name, 'product').values()]
    # Take Indexes for the user negatives using the tokenizer
    kg_pos_tokens_ids = defaultdict(list)
    for head, rel in tqdm(kg_train.keys(), desc="Calculating user negatives tokens ids for link prediction", colour="green"):
        head_token_id, rel_token_id = (tokenizer.convert_tokens_to_ids(f"E{head}") if head not in product_entities
                                       else tokenizer.convert_tokens_to_ids(f"P{head}")), tokenizer.convert_tokens_to_ids(f"R{rel}")
        kg_pos_tokens_ids[(head_token_id,rel_token_id)] = [tokenizer.convert_tokens_to_ids(f"E{tail}") for tail in kg_train[(head,rel)]]
    return all_entities, kg_train, kg_pos_tokens_ids
