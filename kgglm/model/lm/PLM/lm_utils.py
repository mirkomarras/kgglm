import time
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import TrainerCallback

from kgglm.data.dataset.datasets_utils import get_user_negatives
from kgglm.data.knowledge_graph.kg_macros import ENTITY, PRODUCT, RELATION, USER
from kgglm.sampling.samplers.constants import LiteralPath, TypeMapper


def get_user_negatives_and_tokens_ids(
    dataset_name: str, tokenizer
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Returns a dictionary with the user negatives in the dataset, this means the items not interacted in the train and valid sets.
    Note that the ids are the entity ids to be in the same space of the models.
    And a dictionary with the user negatives tokens ids converted
    """
    user_negatives_ids = get_user_negatives(dataset_name)
    user_negatives_tokens_ids = {}
    for uid in tqdm(
        user_negatives_ids.keys(),
        desc="Calculating user negatives tokens ids",
        colour="green",
    ):
        user_negatives_tokens_ids[uid] = [
            tokenizer.convert_tokens_to_ids(f"P{item}")
            for item in user_negatives_ids[uid]
        ]
    return user_negatives_ids, user_negatives_tokens_ids


def _initialise_type_masks(tokenizer, allow_special=False):
    ent_mask = []
    rel_mask = []
    class_weight = 1
    token_id_to_token = dict()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
        # or (not token[0].isalpha() and allow_special) :
        if token[0] == LiteralPath.rel_type:
            rel_mask.append(class_weight)
        else:
            rel_mask.append(0)
        # or (not token[0].isalpha() and allow_special):
        if (
            token[0] == LiteralPath.user_type
            or token[0] == LiteralPath.prod_type
            or token[0] == LiteralPath.ent_type
        ):
            ent_mask.append(class_weight)
        else:
            ent_mask.append(0)

        token_id_to_token[token_id] = token
    return ent_mask, rel_mask, token_id_to_token


def tokenize_augmented_kg(kg, tokenizer, use_token_ids=False):
    type_id_to_subtype_mapping = kg.dataset_info.groupwise_global_eid_to_subtype.copy()
    rel_id2type = kg.rel_id2type.copy()
    type_id_to_subtype_mapping[RELATION] = {int(k): v for k, v in rel_id2type.items()}

    aug_kg = kg.aug_kg

    kg_to_vocab_mapping = dict()
    tokenized_kg = dict()

    for token, token_id in tokenizer.get_vocab().items():
        if not token[0].isalpha():
            continue

        cur_type = token[0]
        cur_id = int(token[1:])

        type = TypeMapper.mapping[cur_type]
        if (
            type == USER
        ):  # Special case since entity ids for user are in a different space compared to the other entities
            subtype = USER
        else:
            subtype = type_id_to_subtype_mapping[type][cur_id]
        if cur_type == LiteralPath.rel_type:
            cur_id = None
        value = token
        if use_token_ids:
            value = token_id
        kg_to_vocab_mapping[(subtype, cur_id)] = value

    for head_type in aug_kg:
        for head_id in aug_kg[head_type]:
            head_key = head_type, head_id
            if head_key not in kg_to_vocab_mapping:
                continue
            head_ent_token = kg_to_vocab_mapping[head_key]
            tokenized_kg[head_ent_token] = dict()

            for rel in aug_kg[head_type][head_id]:
                rel_token = kg_to_vocab_mapping[rel, None]
                tokenized_kg[head_ent_token][rel_token] = set()

                for tail_type in aug_kg[head_type][head_id][rel]:
                    for tail_id in aug_kg[head_type][head_id][rel][tail_type]:
                        tail_key = tail_type, tail_id
                        if tail_key not in kg_to_vocab_mapping:
                            continue
                        tail_token = kg_to_vocab_mapping[tail_key]
                        tokenized_kg[head_ent_token][rel_token].add(tail_token)

    return tokenized_kg, kg_to_vocab_mapping


class TimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []
        self.epoch_start_time = None
        self.in_evaluation = False

    def convert_time(self, sec):
        self.sec = sec
        self.hour = 0
        self.min = 0

        self.sec = self.sec % (24 * 3600)
        self.hour = self.sec // 3600
        self.sec %= 3600
        self.min = self.sec // 60
        self.sec %= 60
        return {
            "hours": int(self.hour),
            "minutes": int(self.min),
            "seconds": int(self.sec),
        }

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_end_time = time.time()
        epoch_duration = self.epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        print(f"Epoch {int(state.epoch)} duration {self.convert_time(epoch_duration)}")


class EmbeddingMapper:
    def __init__(self, tokenizer, kg, embedding):
        self.tokenizer = tokenizer
        self.kg = kg
        self.embedding = embedding

    def get_embedding(self, token_id):
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        if not self.is_kg_element(token):
            return None
        kg_info = self.kg.dataset_info

        element_type = None
        for i, ch in enumerate(token):
            if not ch.isalpha():
                element_type = token[:i]
                break
        element_id = int(token[i:])
        if element_type == "U":
            key = element_id
        elif element_type == "P":
            key = kg_info.groupwise_global_eid_to_cat_eid[PRODUCT][element_id]
        elif element_type == "R":
            key = 0
        else:
            key = kg_info.groupwise_global_eid_to_cat_eid[ENTITY][element_id]

        if element_type == "R":
            emb = self.embedding["rel_embeddings.weight"][key]
        else:
            emb = self.embedding["ent_embeddings.weight"][key]
        return emb

    def is_kg_element(self, token):
        return len(token) > 1 and token[0].isalpha() and token[1:].isnumeric()

    def init_with_embedding(self, wte_weights):
        wte_weights.requires_grad = False
        # elements = set()
        for token, token_id in self.tokenizer.get_vocab().items():
            if self.is_kg_element(token):
                wte_weights[token_id].copy_(
                    torch.FloatTensor(self.get_embedding(token_id).to("cpu"))
                )
