"""PREPROCESSING"""

import numpy as np
import pandas as pd
import os


def get_watched_relation_idx(dataset):
    dataset_path = f"data/{dataset}/preprocessed"
    r_path = f"{dataset_path}/r_map.txt"
    """get relation watched"""
    relation_df = pd.read_csv(r_path, sep="\t")
    # WATCHED = int(relation_df[relation_df['kb_relation'] == 'watched']['id']) # DA SISTEMARE
    WATCHED = relation_df[relation_df['kb_relation']
                          == 'watched']['id'].astype(int).to_list()[0]
    return WATCHED


def dataset_preprocessing(dataset):
    """paths"""

    dataset_path = f"data/{dataset}/preprocessed/"
    preprocessed_torchkge = f"data/{dataset}/preprocessed/torchkge/"
    users_path = f"{dataset_path}users.txt"
    e_new_path = f"{dataset_path}e_map.txt"
    train_path = f"{dataset_path}train.txt"
    valid_path = f"{dataset_path}valid.txt"
    test_path = f"{dataset_path}test.txt"
    i2kg_path = f"{dataset_path}i2kg_map.txt"
    kg_new_final = f"{dataset_path}kg_final.txt"

    WATCHED = get_watched_relation_idx(dataset)
    """create a folder for torchkge data"""
    if not os.path.exists(preprocessed_torchkge):
        os.makedirs(preprocessed_torchkge)

    """Carico utenti ed entità"""
    users_df = pd.read_csv(users_path, sep="\t")['uid']  # uid, gender, age
    e_new_df = pd.read_csv(e_new_path, sep="\t")  # eid, name, entity
    e_last_eid = e_new_df['eid'].iloc[-1]  # prendo l'ultimo elemento

    """prendo gli users e li modifico per concatenarli a e_map"""
    updated_users_to_e_map = []
    for i, uid in enumerate(users_df):
        updated_users_to_e_map.append([i+e_last_eid+1, f"user {uid}", uid])
    updated_users_to_e_map_df = pd.DataFrame(
        updated_users_to_e_map, columns=['eid', 'name', 'entity'])

    """Creo user_mapping.txt"""
    user_mapping = []
    for i, uid in enumerate(users_df):
        user_mapping.append([uid, i])
    user_mapping_df = pd.DataFrame(
        user_mapping, columns=['rating_id', 'new_id'])
    user_mapping_df.to_csv(
        f"{preprocessed_torchkge}user_mapping.txt", sep="\t", index=False)

    """concateno i nuovi utenti formattati adeguatamente alle entità"""
    e_new_df = pd.concat(
        [e_new_df, updated_users_to_e_map_df], ignore_index=True)
    e_new_df.to_csv(
        f"{preprocessed_torchkge}e_map_with_users.txt", sep="\t", index=False)

    """carico train e valid, prendo solo uid e pid"""
    train_df = pd.read_csv(train_path, sep="\t", names=[
                           'uid', 'pid', 'rating', 'timestamp'])[['uid', 'pid']]
    valid_df = pd.read_csv(valid_path, sep="\t", names=[
                           'uid', 'pid', 'rating', 'timestamp'])[['uid', 'pid']]

    """faccio il mapping del pid, quindi creo il mapping tra dataset id ed entity id da i2kg.txt"""
    i2kg_df = pd.read_csv(i2kg_path, sep="\t")
    pid2eid = {pid: eid for pid, eid in zip(
        i2kg_df['pid'], i2kg_df['eid'])}  # pid2id [!]
    entity2eid = {entity: eid for eid, entity in zip(
        e_new_df['eid'], e_new_df['entity'])}  # mapping uid

    """creo le nuove triplette da train con la relazione watched"""
    triplets = [[uid, pid, WATCHED]
                for uid, pid in zip(train_df['uid'], train_df['pid'])]
    triplets_valid = [[uid, pid, WATCHED]
                      for uid, pid in zip(valid_df['uid'], valid_df['pid'])]

    """mappo correttamente i nuovi pid e uid del Train"""
    for i in range(len(triplets)):
        triplets[i][1] = pid2eid[triplets[i][1]]
        triplets[i][0] = entity2eid[triplets[i][0]]

    """mappo correttamente i nuovi pid e uid del Validation"""
    for i in range(len(triplets_valid)):
        triplets_valid[i][1] = pid2eid[triplets_valid[i][1]]
        triplets_valid[i][0] = entity2eid[triplets_valid[i][0]]

    """trasformo triplets e triplets valid in dataframe, quindi salvo le triple ottenute dalla concatenazione tra triplets di train e di valid"""
    triplets_df = pd.DataFrame(
        triplets, columns=['entity_head', 'entity_tail', 'relation'])
    triplets_valid_df = pd.DataFrame(triplets_valid, columns=[
                                     'entity_head', 'entity_tail', 'relation'])
    positives_triplets = pd.concat([triplets_df, triplets_valid_df])
    positives_triplets.to_csv(
        f"{preprocessed_torchkge}triplets_train_valid.txt", sep="\t", index=False)

    """carico il kg e gli concateno le nuove triplette (triplets), quindi salvo le nuove triplette"""
    kg_df = pd.read_csv(kg_new_final, sep="\t")[
        ['entity_head', 'entity_tail', 'relation']]

    triplets_df.to_csv(
        f"{preprocessed_torchkge}triplets_created.txt", sep="\t", index=False)

    kg_updated = pd.concat([kg_df, triplets_df])
    kg_updated.to_csv(
        f"{preprocessed_torchkge}kg_final_updated.txt", sep="\t", index=False)

    """faccio il mapping del test set come ho fatto al train, quindi prendo test.txt, creo nuove triplette con nuova relazione e mappo ai pid e uid corretti"""
    test_df = pd.read_csv(test_path, sep="\t", names=[
                          'uid', 'pid', 'rating', 'timestamp'])
    triplets_test = [[uid, pid, WATCHED]
                     for uid, pid in zip(test_df['uid'], test_df['pid'])]

    for i in range(len(triplets_test)):
        triplets_test[i][0] = entity2eid[triplets_test[i][0]]
        triplets_test[i][1] = pid2eid[triplets_test[i][1]]

    triplets_test = pd.DataFrame(triplets_test)  # non dargli gli header
    triplets_test.to_csv(
        f"{preprocessed_torchkge}triplets_test.txt", sep="\t", index=False)

    # In questo modo sono salvati nel loro ordine originale, contando anche l'aver fatto l'unique
    tmp1, idx1 = np.unique(np.array(triplets_df)[:, 0], return_index=True)
    result = tmp1[np.argsort(idx1)]
    np.save(f"{preprocessed_torchkge}/users_identifiers_new", result)

    tmp1, idx1 = np.unique(np.array(i2kg_df['eid']), return_index=True)
    result = tmp1[np.argsort(idx1)]
    np.save(f"{preprocessed_torchkge}/pids_identifiers_new", result)
