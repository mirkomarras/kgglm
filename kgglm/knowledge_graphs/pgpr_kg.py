from kgglm.knowledge_graphs.kg_utils import (MAIN_PRODUCT_INTERACTION,
                                              get_dataset_relations,
                                              get_entities,
                                              get_entity_tail,
                                              get_knowledge_derived_relations)
from kgglm.knowledge_graphs.kg_base import KnowledgeGraph




class PGPRKnowledgeGraph(KnowledgeGraph):

    def __init__(self, dataset_obj):
        super().__init__()
        self._load_entities(dataset_obj)
        self.dataset_name = dataset_obj.dataset_name
        self._load_reviews(dataset_obj)
        self._load_knowledge(dataset_obj)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities(dataset.dataset_name):
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                relations = get_dataset_relations(dataset.dataset_name, entity)
                self.G[entity][eid] = {r: [] for r in relations}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_reviews(self, dataset):
        print('Load reviews...')

        num_edges = 0
        for rid, data in enumerate(dataset.review.data):
            uid, pid, _, _ = data

            # (2) Add edges.
            main_product, main_interaction = MAIN_PRODUCT_INTERACTION[dataset.dataset_name]
            self._add_edge(USER, uid, main_interaction, main_product, pid)
            num_edges += 2

        print('Total {:d} review edges.'.format(num_edges))

    def _load_knowledge(self, dataset):
        relations = get_knowledge_derived_relations(dataset.dataset_name)
        main_entity, _ = MAIN_PRODUCT_INTERACTION[dataset.dataset_name]
        for relation in relations:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(dataset.dataset_name, relation)
                    self._add_edge(main_entity, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]
