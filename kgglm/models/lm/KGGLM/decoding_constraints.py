import math
from collections import defaultdict

import numpy as np
import torch
from transformers import LogitsProcessor

from kgglm.models.lm.KGGLM.decoding_cache import LFUCache

"""LP logit processor forces last token to reachable ones"""

class ConstrainedLogitsProcessorLP(LogitsProcessor):
    def __init__(self, tokenized_kg, positive_token_map, total_length, tokenizer, num_return_sequences,
                 eos_token_ids, mask_cache_size=3 * 10 ** 4, cand_cache_size=1 * 10 ** 5,
                 **kwargs):
        super().__init__(**kwargs)
        self.kg = tokenized_kg
        self.positive_token_map = positive_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.call_counter_to_process_finished_sequence = 0
        self.eos_token_ids = eos_token_ids
        self.vocab_tokens = [i for i in range(len(self.tokenizer.get_vocab()))]
        self.cache = LFUCache(cand_cache_size)  # dict()
        # self.mask_cache = dict()
        self.mask_cache = LFUCache(mask_cache_size)
        self.special_tokens_ids = [self.tokenizer.encode(x, add_special_tokens=False)[
                                                         0] for x in self.tokenizer.all_special_tokens_extended]

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.total_length:
            # scores[:,[i for i in range(num_tokens) if i not in self.eos_token_ids] ] # float("-Inf") #min_score  # float("-Inf")
            scores[:, :] = -math.inf
            for i in self.eos_token_ids:
                scores[:, i] = 0.
        else:
            for idx in range(scores.shape[0]):
                candidate_tokens = None
                if cur_len % 2 == 1:
                    # parse ent -----> candidate relations
                    k1 = input_ids[idx, -2].item()
                    k2 = input_ids[idx, -1].item()
                    key = k1, k2
                    candidate_tokens = self.positive_token_map[key] + \
                        self.special_tokens_ids
                    scores[idx, candidate_tokens] = -math.inf
        return scores


"""
Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage this means
to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
"""


class ConstrainedLogitsProcessorREC(LogitsProcessor):
    def __init__(self, tokenized_kg, force_token_map, total_length, tokenizer, num_return_sequences,
                 id_to_uid_token_map, eos_token_ids, mask_cache_size=3*10**4, cand_cache_size=1*10**5, **kwargs):
        super().__init__(**kwargs)
        self.kg = tokenized_kg
        self.force_token_map = force_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.id_to_uid_token_map = id_to_uid_token_map
        self.call_counter_to_process_finished_sequence = 0
        self.eos_token_ids = eos_token_ids
        self.vocab_tokens = [i for i in range(len(self.tokenizer.get_vocab()))]
        self.cache = LFUCache(cand_cache_size)  # dict()
        # self.mask_cache = dict()
        self.mask_cache = LFUCache(mask_cache_size)

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        min_score = float("-inf")
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:, :] = -math.inf
            for i in self.eos_token_ids:
                scores[:, i] = 0.
        else:
            mask_list = []

            def init_mask(vocab_size, candidate_tokens):
                    banned_mask = torch.ones(vocab_size, dtype=torch.bool)
                    banned_mask[candidate_tokens] = False
                    return banned_mask

            def convert_iterable(iterable):
                # return torch.LongTensor(list(iterable) ).to(scores.device)
                return list(iterable)

            def cache_ent_rel_cand(key, k1, k2):
                    if k1 in self.kg and k2 in self.kg[k1]:
                        self.cache.put(key,  convert_iterable(self.kg[k1][k2]))
                    else:
                        self.cache.put(key, convert_iterable([]) )
            def cache_rel_cand(key, k1):
                    if k1 in self.kg:
                        self.cache.put(key, convert_iterable(self.kg[k1].keys()))
                    else:
                        self.cache.put(key, convert_iterable([]) )
            # masked_scores = torch.full_like(scores, -math.inf).to(scores.device)
            for idx in range(scores.shape[0]):
                cur_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
                candidate_tokens = None
                if cur_len % 2 == 1:
                    # parse ent -----> candidate relations
                    k1 = input_ids[idx, -2].item()
                    k2 = input_ids[idx, -1].item()
                    key = k1, k2

                    candidate_tokens = self.cache.get(key)
                    if candidate_tokens is None:
                        cache_ent_rel_cand(key, k1, k2)
                    if cur_len == self.total_length - 1:  # Remove from candidates products not in user negatives
                        uid_cond_key = cur_uid, *key
                        candidate_tokens = self.cache.get(uid_cond_key)
                        if candidate_tokens is None:
                            candidates = self.cache.get(key)
                            if candidates is None:
                                cache_ent_rel_cand(key, k1, k2)
                            self.cache.put(uid_cond_key, convert_iterable(
                                set(candidates).intersection(
                                    set(self.force_token_map[cur_uid]))
                            )
                            )
                        key = uid_cond_key
                else:
                    # parse ent->rel    -----> candidates
                    k1 = input_ids[idx, -1].item()
                    key = k1
                    candidate_tokens = self.cache.get(key)
                    if candidate_tokens is None:
                        cache_rel_cand(key, k1)
                candidate_tokens = self.cache.get(key)
                banned_mask = self.mask_cache.get(key)
                if banned_mask is None:
                    banned_mask = np.isin(self.vocab_tokens, candidate_tokens, invert=True)
                    self.mask_cache.put(key, banned_mask)
                mask_list.append(banned_mask)
            banned_tokens_mask = np.vstack(mask_list)  # .to(scores.device)
            scores[banned_tokens_mask] = -math.inf
        return scores

