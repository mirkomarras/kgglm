import collections
import random
from time import time
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp
from helper.datasets.kgatstyle_dataset import KGATStyleDataset


class KGATLoader(KGATStyleDataset):
    def __init__(self, args, path, batch_style='map'):
        super().__init__(args, path, batch_style)
        #self.batch_size_kg = args.batch_size_kg
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()
        self.lap_list = self._get_relational_lap_list()
        self.all_kg_dict = self._get_all_kg_dict()
        self.exist_heads = list(self.all_kg_dict.keys())
        self.N_exist_heads = len(self.exist_heads)
        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(R)
        adj_r_list.append(0)

        adj_mat_list.append(R_inv)
        adj_r_list.append(self.n_relations + 1)
        print('\tconvert ratings into adj mat done.')

        for r_id in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.append(K)
            adj_r_list.append(r_id + 1)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + 2 + self.n_relations)
        print('\tconvert %d relational triples into adj mat done. @%.4fs' % (len(adj_mat_list), time() - t1))

        self.n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):
            relations = [self.adj_r_list[l_id]] * len(lap.row)
            all_kg_dict.update({head: all_kg_dict[head] + [(tail, relation)]
                                for head, tail, relation in zip(lap.row, lap.col, relations)})
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list

    def set_mode(self, mode):
        assert mode in ['cf', 'kg'], "Mode must be either 'cf' or 'kg'"
        self.mode = mode

    def __len__(self) -> int:
        if self.mode == 'cf':
            return self.n_train
        elif self.mode == 'kg':
            return self.N_exist_heads
        else:
            raise ValueError("Invalid mode. Mode must be either 'cf' or 'kg'.")

    def _generate_kge_train_batch(self):
        exist_heads = self.all_kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = random.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [random.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
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
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_users + self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
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


    def __getitem__(self, idx: int) -> Dict[str, Union[int, List[int]]]:
        if self.mode == 'cf':
            u = self.exist_users[idx % self.n_users]
            if u not in self.exist_users:
                raise ValueError(f'Invalid user index {idx}')
            pos_item = self.sample_pos_items_for_u(u, 1)[0]
            neg_item = self.sample_neg_items_for_u(u, 1)[0]
            return {'users': u, 'pos_items': pos_item, 'neg_items': neg_item}
        elif self.mode == 'kg':
            h = self.exist_heads[idx]
            pos_rs, pos_ts = self.sample_pos_triples_for_h(h, 1)
            neg_ts = self.sample_neg_triples_for_h(h, pos_rs[0], 1)
            return {'heads': h, 'relations': pos_rs[0], 'pos_tails': pos_ts[0], 'neg_tails': neg_ts[0]}
        else:
            raise ValueError("Invalid mode. Mode must be either 'cf' or 'kg'.")

    def sample_pos_items_for_u(self, u, num):
        pos_items = self.train_user_dict[u]
        return np.random.choice(pos_items, num, replace=False).tolist()

    def sample_neg_items_for_u(self, u, num):
        user_positives = set(self.train_user_dict[u])
        neg_items = list(self.products.difference(user_positives))
        neg_items = np.random.choice(neg_items, num, replace=False).tolist()
        return neg_items

    def sample_pos_triples_for_h(self, h: int, num: int) -> Tuple[List[int], List[int]]:
        pos_triples = self.all_kg_dict[h]
        sampled_triples = random.sample(pos_triples, num)
        pos_ts, pos_rs = zip(*sampled_triples)
        return list(pos_rs), list(pos_ts)

    def sample_neg_triples_for_h(self, h: int, r: int, num: int) -> List[int]:
        all_possible_tails = set(range(self.n_users + self.n_entities))
        existing_tails = {t for t, rel in self.all_kg_dict[h] if rel == r}
        neg_ts = np.random.choice(list(all_possible_tails - existing_tails), num, replace=False)
        return list(neg_ts)

    def prepare_test_data(self, user_batch, item_batch):
        return {
            'users': user_batch,
            'pos_items': item_batch,
            'mess_dropout': [0.] * len(eval(self.args.layer_size)),
            'node_dropout': [0.] * len(eval(self.args.layer_size)),
        }


