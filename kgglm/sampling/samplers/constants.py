from kgglm.knowledge_graphs.kg_macros import ENTITY, PRODUCT, RELATION, USER


class LiteralPath:
    main_user = ""  #'Us'
    oth_user = ""  #'U'
    ent = ""  #'E'
    prod = ""  #'P'

    user_type = "U"
    prod_type = "P"
    ent_type = "E"
    rel_type = "R"

    recom_prod = ""  #'P'#'Ps'
    fw_rel = ""  #'R' #'Rf'
    back_rel = ""  #'R' #'Rb'
    interaction_rel_id = "-1"

    def get_ids(self, strings):
        TERMINATION = ""
        trie = dict()

        for string in strings:
            cur_trie = trie
            for ch in string:
                if ch not in cur_trie:
                    cur_trie[ch] = dict()
                cur_trie = cur_trie[ch]
            cur_trie[TERMINATION] = ""


class TypeMapper:
    mapping = {
        LiteralPath.user_type: USER,
        LiteralPath.prod_type: PRODUCT,
        LiteralPath.ent_type: ENTITY,
        LiteralPath.rel_type: RELATION,
    }
    inv_mapping = {v: k for k, v in mapping.items()}
