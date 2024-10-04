import os
import json

current_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_path, 'config_files', 'kg_structure_config.json'), 'r') as kg_config_structure_file:
    kgs_conf_structure = json.load(kg_config_structure_file)
with open(os.path.join(current_path, 'config_files', 'kg_metapaths_config.json'), 'r') as kg_config_metapaths_file:
    kgs_conf_metapaths = json.load(kg_config_metapaths_file)

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

USER = "user"
PRODUCT = "product"
ENTITY = "entity"
RELATION = "relation"
SELF_LOOP = "self_loop"
REV_PREFIX = "rev_"

INTERACTION = {
    dset: list(kg_conf[USER].keys())[0] for dset, kg_conf in kgs_conf_structure.items()
}
MAIN_PRODUCT_INTERACTION = {
    dset: (PRODUCT, inter) for dset, inter in INTERACTION.items()
}
ENTITY_LIST = {
    dset: list(kg_conf.keys()) for dset, kg_conf in kgs_conf_structure.items()
}
RELATION_LIST = {
    dset: sum((list(kg_conf[et].keys()) for et in kg_conf), start=[]) for dset, kg_conf in kgs_conf_structure.items()
}

REV_INTERACTION = {
    dset: REV_PREFIX + dset_interaction for dset, dset_interaction in INTERACTION.items()
}
REV_RELATIONS = {
    dset: [REV_PREFIX + rel for rel in dset_relations] for dset, dset_relations in RELATION_LIST.items()
}

TRANSE='TransE'

# UCPR SPECIFIC RELATIONs
KG_RELATION = kgs_conf_structure

# 0 is reserved to the main relation, 1 to mention
PATH_PATTERN = kgs_conf_metapaths
