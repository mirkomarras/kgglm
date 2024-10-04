import gzip
import os
from collections import defaultdict

import numpy as np
from easydict import EasyDict as edict

from kgglm.knowledge_graphs.kg_macros import ENTITY, PRODUCT, USER
from kgglm.utils import get_dataset_info_dir

VERBOSE = False

class KARSDataset(object):
    """This class is used to load data files and save in the instance.
    This dataset is used for the following models: {PGPR, UCPR}
    """

    def __init__(self, dataset_name, set_name='train', data_dir=None):
        self.dataset_name = dataset_name
        if data_dir is None:
            self.data_dir = get_dataset_info_dir(self.dataset_name)
        else:
            self.data_dir = data_dir
        self.groupwise_global_eid_to_cat_eid = {}
        self.groupwise_global_eid_to_subtype = {}
        self.global_eid_to_cat_eid = {}
        self.cat_eid_to_global_eid = {}
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        self.entity_filename_edict, self.relation_filename_edict = self.infer_kg_structure()
        #Other relation names doesn't contain the main interaction
        self.entity_names, self.other_relation_names = list(self.entity_filename_edict.keys()), list(self.relation_filename_edict.keys())
        self.load_entities(self.entity_filename_edict)
        self.load_product_relations(self.relation_filename_edict)
        self.load_reviews()

    #This will not work if your entities are composed by multiple words, e.g. if you name an entity related_product
    #this script will consider as a relation, please use a single word for relations
    def infer_kg_structure(self):
        file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.txt.gz')]
        entity_filenames = [filename for filename in file_list if len(filename.split("_")) == 1]

        entity_filename_edict = edict()
        entity_names = []
        for entity_file in entity_filenames:
            if os.path.isdir(os.path.join(self.data_dir, entity_file)): continue
            name = entity_file.split(".")[0]
            if name in ["train", "valid", "test"]: continue
            entity_names.append(name)
            entity_filename_edict[name] = entity_file

        relation_filenames = [filename for filename in file_list if len(filename.split("_")) > 1]
        
        relation_filename_edict = edict()
        relation_names = []
        for relation_file in relation_filenames:
            name = relation_file.split(".")[0]
            relation_names.append(name)
            relation_filename_edict[name] = relation_file

        self.relation2entity = {}
        for rel_name in relation_names:
            entity_name = rel_name.split("_")[-1]
            self.relation2entity[rel_name] = entity_name

        return entity_filename_edict, relation_filename_edict

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self, entity_filename_edict):
        """Load 10 global entities from data files:
        'user','movie','actor','director','producer','production_company','category','editor','writter','cinematographer'
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        
        for name in entity_filename_edict:
            vocab = []
            self.global_eid_to_cat_eid[name] = dict()
            self.cat_eid_to_global_eid[name] = dict()
            # remove header with [1:]
            for x in self._load_file(entity_filename_edict[name])[1:]:
                cat_eid, global_eid = x.split("\t")
                vocab.append(cat_eid)
                global_eid = int(global_eid)
                cat_eid = int(cat_eid)
                group_id = None
                if name == PRODUCT:
                    global_eid = cat_eid
                    group_id = PRODUCT
                elif name == USER:
                    group_id = USER
                else:
                    group_id = ENTITY
                if group_id not in self.groupwise_global_eid_to_cat_eid:
                    self.groupwise_global_eid_to_cat_eid[group_id] = dict()
                    self.groupwise_global_eid_to_subtype[group_id] = dict()
                self.groupwise_global_eid_to_cat_eid[group_id][global_eid] = cat_eid
                self.groupwise_global_eid_to_subtype[group_id][global_eid] = name

                self.global_eid_to_cat_eid[name][global_eid] = cat_eid
                if name != PRODUCT and name != USER:
                    if ENTITY not in self.global_eid_to_cat_eid:
                        self.global_eid_to_cat_eid[ENTITY] = dict()
                    self.global_eid_to_cat_eid[ENTITY][global_eid] = cat_eid
                self.cat_eid_to_global_eid[name][cat_eid] = global_eid
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab) + 1))
            if VERBOSE:
                print('Load', name, 'of size', len(vocab))

    def load_product_relations(self, relation_filename_edict):
        """Load 8 product -> ? relations:
        - 'directed_by': movie -> director
        - 'produced_by_company': movie->production_company,
        - 'produced_by_producer': movie->producer,
        - 'starring': movie->actor,
        - 'belong_to': movie->category,
        - 'edited_by': movie->editor,
        - 'written_by': movie->writter,
        - 'cinematography': movie->cinematographer,

        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """

        product_relations = edict()
        for rel_name, rel_filename in relation_filename_edict.items():
            entity_name = self.relation2entity[rel_name]
            product_relations[rel_name] = (rel_filename, getattr(self, entity_name))

        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                data=[],
                et_vocab=product_relations[name][1].vocab,  # copy of brand, catgory ... 's vocab
                et_distrib=np.zeros(product_relations[name][1].vocab_size)  # [1] means self.brand ..
            )
            size = 0
            for line in self._load_file(product_relations[name][0]):  # [0] means brand_p_b.txt.gz ..
                knowledge = []
                line = line.split('\t')
                for x in line:  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                        size += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            if VERBOSE:
                print('Load', name, 'of size', size)

        all_kg_entities = set()
        pid_to_kg_entities_negative_samples = dict()
        pid_to_kg_entities = defaultdict(set)
        kg_entities_to_pid = defaultdict(set)
        kg_entities_to_pid_negative_samples = dict()

        self.unique_relations = set()
        for rel_name in product_relations:
            self.unique_relations.add(rel_name)
            relation_edict = getattr(self, rel_name)
            for pid, kg_entities in enumerate(relation_edict.data):
                pid_to_kg_entities[(pid, rel_name)] = set(kg_entities)
                for kg_ent in kg_entities:
                    kg_entities_to_pid[(kg_ent, rel_name)].add(pid)
                all_kg_entities.update(kg_entities)

        for (pid, rel_name), kg_entities in pid_to_kg_entities.items():
            pid_to_kg_entities_negative_samples[(pid, rel_name)] = list(all_kg_entities - kg_entities)

        products = set( [ k1 for k1,k2 in  pid_to_kg_entities.keys()] )
        for (kg_ent, rel_name), pids in kg_entities_to_pid.items():
            kg_entities_to_pid_negative_samples[(kg_ent, rel_name)] = list(products - pids)
            
        self.pid_to_kg_entities = pid_to_kg_entities
        self.pid_to_kg_entities_negative_samples = pid_to_kg_entities_negative_samples
        self.all_kg_entities = all_kg_entities
        self.kg_entities_to_pid = kg_entities_to_pid
        self.kg_entities_to_pid_negative_samples = kg_entities_to_pid_negative_samples


    def load_reviews(self):
        """Load user-product reviews from train/test data files.
            Create member variable `review` associated with following attributes:
            - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
            - `size`: number of reviews.
            - `product_distrib`: product vocab frequency among all eviews.
            - `product_uniform_distrib`: product vocab frequency (all 1's)
            - `word_distrib`: word vocab frequency among all reviews.
            - `review_count`: number of words (including duplicates).
            - `review_distrib`: always 1.
            """
        
        review_data = []  # (user_idx, product_idx, rating out of 5, timestamp)
        #product_distrib = np.zeros(self.product.vocab_size)
        invalid_users = 0
        invalid_pid = 0

        self.users = set()
        self.products = set()
        self.user_pids = defaultdict(set)
        self.pid_users = defaultdict(set)
        self.user_negative_pids = dict()
        self.pid_negative_users = dict()

        for line in self._load_file(self.review_file):
                arr = line.split('\t')
                user_idx = int(arr[0])
                product_idx = int(arr[1])
                rating = int(arr[2])
                timestamp = int(arr[3])
                review_data.append((user_idx, product_idx, rating, timestamp))
                self.user_pids[user_idx].add(product_idx)
                self.pid_users[product_idx].add(user_idx)
                self.users.add(user_idx)
                self.products.add(product_idx)
                #product_distrib[product_idx] += 1

        for uid in self.users:
            self.user_negative_pids[uid] = list(self.products - self.user_pids[uid])
        for pid in self.products:
            self.pid_negative_users[pid] = list(self.users - self.pid_users[pid])

        if VERBOSE:
            print(f"Invalid users: {invalid_users}, invalid items: {invalid_pid}")

        self.review = edict(
            data=review_data,
            size=len(review_data),
            #product_distrib=product_distrib,
            #product_uniform_distrib=np.ones(self.product.vocab_size),
            review_count=len(review_data),
            review_distrib=np.ones(len(review_data))  # set to 1 now
        )
        if VERBOSE:
            print('Load review of size', self.review.size)
