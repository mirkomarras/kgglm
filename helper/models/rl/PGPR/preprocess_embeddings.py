import argparse
import glob
import os
import pickle
from collections import defaultdict

import torch

from helper.datasets.KARSDataset import KARSDataset
from helper.knowledge_graphs.kg_macros import LFM1M, ML1M, PRODUCT, USER
from helper.models.kge.utils import get_weight_ckpt_dir
from helper.utils import (check_dir, get_data_dir, get_dataset_info_dir,
                          get_model_dir)


class FormatEmbeddings(KARSDataset):
    def __init__(self, args):
        super().__init__(args.dataset)
        self.name=args.name
        self.dataset_name=args.dataset
        self.weight_dir_checkpoints=get_weight_ckpt_dir(self.name,self.dataset_name)

        self._integrity_checks()

        self.embeddings=self._load_latest_embedding()
        self.ent_mappings=self._create_entity_mappings()
        self.rel_mappings = self._create_relation_mappings()

        self._create_new_embeddings(self.ent_mappings,self.rel_mappings,self.embeddings)

    def _integrity_checks(self):
        assert self.dataset_name in [LFM1M,ML1M], "Insert one of ML1M or LFM1M in uppercase"
        # todo: checks on valid name based on models inside helepr/models/kge/

    def _load_latest_embedding(self):
        checkpoints = glob.glob(os.path.join(self.weight_dir_checkpoints, f'{self.name}_dataset_{self.dataset_name}_*.pth'))
        assert len(checkpoints)!=0, "Error: Please train first the kge method!"
        ckpt=max(checkpoints, key=os.path.getmtime) # get latest ckpt
        return torch.load(ckpt)

    def _create_entity_mappings(self):
        mapping=defaultdict(list)
        user2eid = self._user2eid()
        product2eid = self._product2eid()
        for entityType in self.entity_filename_edict.keys():
            for i,x in enumerate(self._load_file(self.entity_filename_edict[entityType])):
                if i==0:
                    # skip header line
                    continue
                cat_id, global_eid=x.split('\t')
                if entityType==USER:
                    mapping[entityType].append(user2eid[int(global_eid)])
                elif entityType==PRODUCT:
                    mapping[entityType].append(product2eid[int(global_eid)])
                else:
                    mapping[entityType].append(int(global_eid))
        return mapping

    def _load_r_map(self):
        with open(os.path.join(get_data_dir(self.dataset_name),'r_map.txt'),'r') as f:
            return [line.strip('\t') for line in f]

    def _create_relation_mappings(self):
        mapping=defaultdict(list)
        relations=self._load_r_map()
        for i,relation in enumerate(relations):
            if i==0:
                # skip header line
                continue
            rid, _, r_name = relation.split('\t')
            # necessary bypass because of r_map on lfm1m
            #r_name=r_name.split("_")[0]
            if self.dataset_name==LFM1M and r_name == 'watched':
                mapping['listened'].append(int(rid))
            else:
                mapping[r_name.split("\n")[0]].append(int(rid))
        return mapping

    def _create_new_embeddings(self,ent_mappings, rel_mappings, embeddings):
        # assuming we'll always have ent_embeddings and rel_embeddings.
        ent_embeddings=embeddings['ent_embeddings.weight']
        rel_embeddings = embeddings['rel_embeddings.weight']

        embeds=dict()
        for relation in rel_mappings.keys():
            embeds[relation]=rel_embeddings[rel_mappings[relation]]

        for entity in ent_mappings.keys():
            embeds[entity]=ent_embeddings[ent_mappings[entity]]

        # save
        check_dir(os.path.join(get_model_dir('PGPR','rl'),'embeddings'))
        embed_file = os.path.join(get_model_dir('PGPR','rl'),'embeddings', f'{self.name}_structured_{self.dataset_name}.pkl')
        pickle.dump(embeds, open(embed_file, 'wb'))

    def _user2eid(self):
        mapping=dict()
        mapping_folder = get_dataset_info_dir(self.dataset_name)
        with open(os.path.join(mapping_folder,'user_mapping.txt'),'r') as f:
           for i,line in enumerate(f):
               if i==0:
                   # skip header line
                   continue
               rating_id, eid = line.split('\t')
               mapping[int(rating_id)]=int(eid)
        return mapping


    def _product2eid(self):
        mapping = dict()
        mapping_folder = get_dataset_info_dir(self.dataset_name)
        with open(os.path.join(mapping_folder, 'product_mapping.txt'), 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # skip header line
                    continue
                rating_id, eid = line.split('\t')
                mapping[int(rating_id)] = int(eid)
        return mapping

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M,
                        help='Currently supported: {ml1m | lfm1m}.')
    parser.add_argument('--name', type=str,
                        default='TransE', help='kge model name.')
    args,unk = parser.parse_known_args()

    FormatEmbeddings(args)