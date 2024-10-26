import math
from collections import defaultdict
from typing import Literal, Union

import numpy as np
import torch
from transformers import LogitsProcessor

from kgglm.model.decoding_utils import LFUCache


class ConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    """
    Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage
    this means to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
    If task is link prediction (LP) logit processor forces last token to reachable ones
    """

    RECOMMENDATION_TASK = "recommendation"
    LINK_PREDICTION_TASK = "link_prediction"

    def __init__(
        self,
        tokenized_kg,
        force_token_map,
        total_length,
        tokenizer,
        num_return_sequences,
        id_to_uid_token_map,
        eos_token_ids,
        mask_cache_size=3 * 10**4,
        cand_cache_size=1 * 10**5,
        task: Union[
            Literal["recommendation"], Literal["link_prediction"]
        ] = "recommendation",
        **kwargs,
    ):
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
        self.cache = LFUCache(cand_cache_size)
        self.mask_cache = LFUCache(mask_cache_size)
        self.task = task

        if self.task == self.LINK_PREDICTION_TASK:
            self.special_tokens_ids = [
                self.tokenizer.encode(x, add_special_tokens=False)[0]
                for x in self.tokenizer.all_special_tokens_extended
            ]
        else:
            self.special_tokens_ids = None

    def cache_candidate(self, cache_key, key1, key2=None):
        """
        :param cache_key:
        :param key1:
        :param key2: if key2 != None, it is used to cache an entity, otherwise a relation
        """
        if key1 in self.kg:
            if key2 is not None and key2 in self.kg[key1]:
                self.cache.put(cache_key, list(self.kg[key1][key2]))
            else:
                self.cache.put(cache_key, list(self.kg[key1].keys()))
        else:
            self.cache.put(cache_key, list([]))

    def mask_non_eos_tokens(self, scores):
        """Apply masking to all tokens except EOS tokens."""
        scores[:, :] = -math.inf
        for i in self.eos_token_ids:
            scores[:, i] = 0.0

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        if current_len == self.total_length:
            self.mask_non_eos_tokens(scores)
        else:
            mask_list = []

            for idx in range(scores.shape[0]):
                if self.task == self.RECOMMENDATION_TASK:
                    key, candidate_tokens = self.process_scores_rec(
                        input_ids, idx, current_len
                    )
                    banned_mask = self.get_banned_mask(key, candidate_tokens)
                    mask_list.append(banned_mask)
                elif self.task == self.LINK_PREDICTION_TASK:
                    candidate_tokens = self.process_scores_lp(
                        input_ids, idx, current_len
                    )
                    if candidate_tokens is not None:
                        scores[idx, candidate_tokens] = -math.inf

            if self.task == self.RECOMMENDATION_TASK:
                banned_tokens_mask = np.vstack(mask_list)
                scores[banned_tokens_mask] = -math.inf

        return scores

    def process_scores_rec(self, input_ids, idx, current_len):
        """Process each score based on input length and update mask list."""
        key = self.get_current_key(input_ids, idx, current_len)
        if current_len % 2 == 1:
            current_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
            candidate_tokens = self.get_candidates_rec(key, current_len, current_uid)
        else:
            candidate_tokens = self.cache.get(key) or self.cache_candidate(key, key)

        return key, candidate_tokens

    def process_scores_lp(self, input_ids, idx, current_len):
        """Process each score based on input length or skip."""
        candidate_tokens = None
        if current_len % 2 == 1:
            key = self.get_current_key(input_ids, idx, 1)
            candidate_tokens = self.get_candidates_lp(key)

        return candidate_tokens

    @staticmethod
    def get_current_key(input_ids, idx, current_len):
        if current_len % 2 == 1:
            return input_ids[idx, -2].item(), input_ids[idx, -1].item()
        else:
            return input_ids[idx, -1].item()

    def get_candidates_rec(self, key, current_len, current_uid):
        """Retrieve candidate tokens and update cache based on current length."""
        candidate_tokens = self.cache.get(key)
        if candidate_tokens is None:
            self.cache_candidate(key, *key)

        if current_len == self.total_length - 1:
            uid_cond_key = (current_uid, *key)
            if self.cache.get(uid_cond_key) is None:
                candidate_tokens = set(self.cache.get(key))

                current_uid_forced_tokens = set(self.force_token_map[current_uid])
                candidate_tokens = candidate_tokens.intersection(
                    current_uid_forced_tokens
                )
                self.cache.put(uid_cond_key, list(candidate_tokens))
            key = uid_cond_key

        return self.cache.get(key)

    def get_candidates_lp(self, key):
        return self.force_token_map[key] + self.special_tokens_ids

    def get_banned_mask(self, key, candidate_tokens):
        """Retrieve or cache the banned token mask for a specific key."""
        banned_mask = self.mask_cache.get(key)
        if banned_mask is None:
            banned_mask = np.isin(self.vocab_tokens, candidate_tokens, invert=True)
            self.mask_cache.put(key, banned_mask)
        return banned_mask


class PrefixConstrainedLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    def __init__(
        self,
        tokenized_kg,
        force_token_map,
        total_length,
        tokenizer,
        num_return_sequences,
        id_to_uid_token_map,
        eos_token_ids,
        mask_cache_size=3 * 10**4,
        cand_cache_size=1 * 10**5,
        **kwargs,
    ):
        super().__init__(
            tokenized_kg,
            force_token_map,
            total_length,
            tokenizer,
            num_return_sequences,
            id_to_uid_token_map,
            eos_token_ids,
            mask_cache_size=mask_cache_size,
            cand_cache_size=cand_cache_size,
            **kwargs,
        )
        self.mask_cache = None

    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[-1]
        if current_len == self.total_length:
            self.mask_non_eos_tokens(scores)
        else:
            indices = []
            masked_scores = torch.full_like(scores, -math.inf)
            for idx in range(scores.shape[0]):
                _, candidate_tokens = self.process_scores(input_ids, idx, current_len)

                candidate_tokens = torch.LongTensor(
                    candidate_tokens, device=scores.device
                )
                indices.append(candidate_tokens)
                masked_scores[idx].scatter_(
                    dim=-1, index=candidate_tokens, src=scores[idx]
                )
            scores = masked_scores

        return scores


class PLMLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    """
    https://dl.acm.org/doi/pdf/10.1145/3485447.3511937
    Constraint decoding strategy for PLM, it forces the model to generate alternatively entities and relations
    """

    def __init__(
        self,
        tokenized_kg,
        force_token_map,
        total_length,
        tokenizer,
        num_return_sequences,
        id_to_uid_token_map,
        eos_token_ids,
        ent_mask,
        rel_mask,
        token_id_to_token,
        mask_cache_size=3 * 10**4,
        cand_cache_size=1 * 10**5,
        **kwargs,
    ):
        super().__init__(
            tokenized_kg,
            force_token_map,
            total_length,
            tokenizer,
            num_return_sequences,
            id_to_uid_token_map,
            eos_token_ids,
            mask_cache_size=mask_cache_size,
            cand_cache_size=cand_cache_size,
            **kwargs,
        )
        self.ent_ids = [idx for idx, elem in enumerate(ent_mask) if elem > 0]
        self.rel_ids = [idx for idx, elem in enumerate(rel_mask) if elem > 0]

        self.ent_mask = [elem > 0 for idx, elem in enumerate(ent_mask)]
        self.rel_mask = [elem > 0 for idx, elem in enumerate(rel_mask)]

        self.token_id_to_token = token_id_to_token

    @staticmethod
    def get_current_key(input_ids, idx, current_len):
        return int(current_len % 2 == 1)

    def process_scores(self, input_ids, idx, current_len):
        """Process each score based on input length and update mask list."""
        key = self.get_current_key(input_ids, idx, current_len)
        current_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
        return self.get_key_and_candidates(key, idx, current_len, current_uid)

    def get_key_and_candidates(self, key, idx, current_len, current_uid):
        """Retrieve candidate tokens and update key based on current length."""
        if current_len % 2 == 1:
            candidate_tokens = self.ent_ids

            if current_len == self.total_length - 1:
                candidate_tokens = self.force_token_map[current_uid]
                key = current_uid, idx
        else:
            candidate_tokens = self.rel_ids
            key = 0

        return key, candidate_tokens
