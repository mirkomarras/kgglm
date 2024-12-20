import argparse
import random

import numpy as np

from helper.knowledge_graphs.kg_macros import PRODUCT, USER
from helper.knowledge_graphs.kg_utils import KG_RELATION
from helper.models.rl.CAFE.cafe_utils import load_kg, load_user_products


class ReplayMemory(object):
    def __init__(self, memory_size=5000):
        self.memory_size = memory_size
        self.memory = []

    def add(self, data):
        # `data` is a list of objects.
        self.memory.extend(data)
        while len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def sample(self):
        if not self.memory:  # memory is empty.
            return None
        return random.choice(self.memory)

    def __len__(self):
        return len(self.memory)


class OnlinePathLoader:
    def __init__(self, dataset, batch_size, topk=20):
        self.kg = load_kg(dataset)  # KnowledgeGraph
        self.num_users = len(self.kg.get(USER))
        self.num_products = len(self.kg.get(PRODUCT))
        self.mpath_ids = list(range(len(self.kg.metapaths)))  # metapath IDs
        self.topk = topk
        self.topk_user_products = load_user_products(dataset, 'pos')[:, :self.topk]
        self.batch_size = batch_size
        self.data_size = self.num_users * len(self.mpath_ids) * self.topk
        self.total_steps = int(self.data_size / self.batch_size)

        self.memory_size = 10000  # number of paths to save for each metapath
        self.replay_memory = {}
        for mpid in self.mpath_ids:
            self.replay_memory[mpid] = ReplayMemory(self.memory_size)

        self._steps = 0
        self._has_next = True
        self.reset()

    def reset(self):
        self._steps = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        # Uniformly sample a metapath
        mpid = np.random.choice(self.mpath_ids)

        pos_path_batch, neg_pid_batch = [], []
        it = 0
        while len(pos_path_batch) < self.batch_size:
            # Sample a user and a good product.
            uid = np.random.choice(self.num_users)
            # pid = np.random.choice(self.topk_user_products[uid])
            pidx = np.random.choice(self.topk)
            pid = self.topk_user_products[uid][pidx]

            # Compute the probability to sample path from memory, P \in [0, 0.5].
            use_memory_prob = 0.5 * len(self.replay_memory[mpid]) / self.memory_size

            # Sample a history path from memory.
            if np.random.rand() < use_memory_prob:
                hist_path = self.replay_memory[mpid].sample()
                if hist_path is None:  # no memory
                    continue
                pos_path_batch.append(hist_path)

            # Sample a new path from graph.
            else:
                paths = self.kg.fast_sample_path_with_target(mpid, uid, pid, 1)
                if not paths:  # no path is found
                    continue
                pos_path_batch.append(paths[0])
                self.replay_memory[mpid].add(paths)

            # Sample a negative product.
            if pidx < self.topk - 1:
                neg_pidx = np.random.choice(np.arange(pidx + 1, self.topk))
                neg_pid = self.topk_user_products[uid][neg_pidx]
            else:
                neg_pid = np.random.choice(self.num_products)
            neg_pid_batch.append(neg_pid)

        pos_path_batch = np.array(pos_path_batch)
        neg_pid_batch = np.array(neg_pid_batch)

        self._steps += 1
        self._has_next = self._steps < self.total_steps

        return mpid, pos_path_batch, neg_pid_batch

class KGMask(object):
    def __init__(self, kg):
        self.kg = kg

    def get_ids(self, eh, eh_ids, relation):
        et_ids = []
        if isinstance(eh_ids, list):
            for eh_id in eh_ids:
                ids = list(self.kg(eh, eh_id, relation))
                et_ids.extend(ids)
            et_ids = list(set(et_ids))
        else:
            et_ids = list(self.kg(eh, eh_ids, relation))
        return et_ids

    def get_mask(self, eh, eh_ids, relation):
        et = KG_RELATION[eh][relation]
        et_vocab_size = len(self.kg(et))

        if isinstance(eh_ids, list):
            mask = np.zeros([len(eh_ids), et_vocab_size], dtype=np.int64)
            for i, eh_id in enumerate(eh_ids):
                et_ids = list(self.kg(eh, eh_id, relation))
                mask[i, et_ids] = 1
        else:
            mask = np.zeros(et_vocab_size, dtype=np.int64)
            et_ids = list(self.kg(eh, eh_ids, relation))
            mask[et_ids] = 1
        return mask

    def get_et(self, eh, eh_id, relation):
        return np.array(self.kg(eh, eh_id, relation))

    def __call__(self, eh, eh_ids, relation):
        return self.get_mask(eh, eh_ids, relation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beauty', help='One of {ml1m | lfm1m}')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    args = parser.parse_args()