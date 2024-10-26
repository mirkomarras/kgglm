import random as rd
from typing import Dict, List

import numpy as np

from kgglm.datasets.kgatstyle_dataset import KGATStyleDataset


class CKELoader(KGATStyleDataset):
    def __init__(self, args, path: str, batch_style: str = "map"):
        super().__init__(args, path, batch_style)
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        kg_file = path + "/kg_final.txt"
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

    def _generate_kge_train_batch(self):
        # triple. dict(head:[(tail1, rel1),(tail2 , rel2), ecc])
        exist_heads = self.kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num:
                    break

                t = np.random.randint(low=0, high=self.n_entities, size=1)[0]
                if (t, r) not in self.kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            return np.random.choice(pos_items, num, replace=False).tolist()

        def sample_neg_items_for_u(u, num):
            user_positives = set(self.train_user_dict[u])
            neg_items = list(self.products.difference(user_positives))
            neg_items = np.random.choice(neg_items, num, replace=False).tolist()
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def generate_train_batch(self) -> Dict[str, List[int]]:
        users, pos_items, neg_items = self._generate_train_cf_batch()
        heads, relations, pos_tails, neg_tails = self._generate_kge_train_batch()

        return {
            "users": users,
            "pos_items": pos_items,
            "neg_items": neg_items,
            "heads": heads,
            "relations": relations,
            "pos_tails": pos_tails,
            "neg_tails": neg_tails,
        }

    def prepare_test_data(self, user_batch, item_batch):
        return {
            "users": user_batch,
            "pos_items": item_batch,
        }
