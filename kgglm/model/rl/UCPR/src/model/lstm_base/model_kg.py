from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from kgglm.datasets.datasets_utils import load_dataset, load_kg
from kgglm.knowledge_graphs.kg_macros import INTERACTION, SELF_LOOP
from kgglm.model.rl.UCPR.utils import PADDING, TMP_DIR

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class KG_KGE(nn.Module):
    def __init__(self, args):
        super(KG_KGE, self).__init__()
        self.embed_size = args.embed_size
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        dataset = load_dataset(args.dataset, TMP_DIR)
        self.kg = load_kg(args.dataset, TMP_DIR)
        self.entities = edict()
        self.requires_grad = args.kg_emb_grad
        self.dataset_name = args.dataset
        self.relation_names = dataset.other_relation_names
        self.entity_names = dataset.entity_names
        self.relation2entity = dataset.relation2entity

        # Initialize entity embeddings.
        self.initialize_entity_embeddings(dataset)

        for e in self.entities:
            embed = self._entity_embedding(e, self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.initialize_relations_embeddings(dataset)
        for r in self.relations:
            embed = self._relation_embedding(r)
            setattr(self, r, embed)
            # bias = self._relation_bias(len(self.relations[r].et_distrib))
            # setattr(self, r + '_bias', bias)
        embed = self._relation_embedding(PADDING)
        setattr(self, PADDING, embed)
        embed = self._relation_embedding(SELF_LOOP)
        setattr(self, SELF_LOOP, embed)

    def initialize_entity_embeddings(self, dataset):
        self.entities = edict()
        for entity_name in self.entity_names:
            value = edict(vocab_size=getattr(dataset, entity_name).vocab_size)
            self.entities[entity_name] = value

    def initialize_relations_embeddings(self, dataset):
        self.relations = edict()
        for relation_name in dataset.other_relation_names:
            value = edict(
                et=dataset.relation2entity[relation_name],
                # et_distrib=self._make_distrib(getattr(dataset, relation_name).et_distrib)
            )
            self.relations[relation_name] = value
        main_rel = INTERACTION[dataset.dataset_name]
        self.relations[main_rel] = edict(
            et="product",
            # et_distrib=self._make_distrib(getattr(dataset, "review").product_uniform_distrib)
        )

    def _entity_embedding(self, key, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
        Note that last dimension is always 0's.
        """
        embed = nn.Embedding(
            vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False
        )
        embed.weight.requires_grad = self.requires_grad
        return embed

    def _relation_embedding(self, key):
        """Create relation vector of size [1, embed_size]."""
        weight = torch.randn(1, self.embed_size, requires_grad=True)
        embed = nn.Parameter(weight[:, : self.embed_size])
        embed.requires_grad = self.requires_grad

        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float64), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def lookup_emb(self, node_type, type_index):
        embedding_file = getattr(self, node_type)
        entity_vec = embedding_file(type_index)
        return entity_vec

    def lookup_rela_emb(self, node_type):
        relation_vec = getattr(self, node_type)
        return relation_vec
