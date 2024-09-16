import json
from pathlm.knowledge_graphs.kg_macros import *

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

def get_KG_structures():
    with open(os.path.join(ROOT_DIR, 'helper/knowledge_graphs/config_files/kg_structure_config.json'), 'r') as f:
        KG_STRUCTURES = json.load(f)
    return KG_STRUCTURES

# 0 is reserved to the main relation, 1 to mention
KG_RELATION = get_KG_structures()

MAIN_PRODUCT_INTERACTION = {
    ML1M: (PRODUCT, INTERACTION[ML1M]),
    LFM1M: (PRODUCT, INTERACTION[LFM1M])
}

