from collections import defaultdict

import torch


def normalize_tuple(logits_tuple):
    # Normalize each tensor in the tuple
    normalized_tuple = tuple(torch.softmax(logits, dim=-1)
                             for logits in logits_tuple)
    return normalized_tuple


class RankerLP():
    def __init__(self, tokenizer, kg_positives, K=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.kg_positives = kg_positives
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)
        self.max_new_tokens = max_new_tokens
        self.K = K

    def update_topk(self, generate_outputs):
        sorted_scores = generate_outputs.sequences_scores.argsort(
            descending=True)
        generate_outputs.sequences = generate_outputs.sequences[sorted_scores]
        for sequence in generate_outputs.sequences:
            sequence = self.tokenizer.decode(sequence).split(' ')
            head_eid = int(sequence[1][1:])
            rel_rid = int(sequence[2][1:])
            if len(self.topk[head_eid, rel_rid]) >= self.K:
                continue
            recommended_token = sequence[-1]
            recommended_item = int(recommended_token[1:])
            if recommended_item in self.kg_positives[(head_eid, rel_rid)] or recommended_item in self.topk[head_eid, rel_rid]:
                continue
            self.topk[head_eid, rel_rid].append(recommended_item)
            self.topk_sequences[head_eid, rel_rid].append(sequence)

    def reset_topks(self):
        del self.topk
        del self.topk_sequences
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)


class CumulativeSequenceScoreRanker():
    def __init__(self, tokenizer, user_negatives, K=10, max_new_tokens=24):
        self.tokenizer = tokenizer
        self.user_negatives = user_negatives
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)
        self.max_new_tokens = max_new_tokens
        self.K = K
        # Can be changed accordingly to what you want to do
        self.sequence_scorer_fnc = self.calculate_sequence_scores

    def calculate_sequence_scores(self, normalized_tuple, sequences):
        # print('aaaa')
        # print(sequences.shape, self.max_new_tokens)
        # print(len(normalized_tuple), normalized_tuple[0].shape)
        last_5_tokens = sequences[:, -self.max_new_tokens:]
        sequence_scores = []
        # Iterate over each tensor in the normalized tuple
        for i in range(self.max_new_tokens):
            # Get the probabilities corresponding to the ith token in last_5_tokens
            probs = normalized_tuple[i].gather(
                1, last_5_tokens[:, i].unsqueeze(1))
            sequence_scores.append(probs)
        # Convert the list of tensors into a single tensor
        sequence_scores = torch.cat(sequence_scores, dim=-1)
        # Calculate the average score over the last 5 positions for each sequence
        sequence_scores = sequence_scores.mean(dim=-1)
        return sequence_scores

    def update_topk(self, generate_outputs):
        generate_outputs.scores = normalize_tuple(generate_outputs.scores)
        generate_outputs.sequences_scores = self.sequence_scorer_fnc(
            generate_outputs.scores, generate_outputs.sequences)
        sorted_indices = generate_outputs.sequences_scores.argsort(
            descending=True)
        sorted_sequences = generate_outputs.sequences[sorted_indices]

        for sequence in sorted_sequences:

            sequence = self.tokenizer.decode(sequence).split(' ')

            uid_token = sequence[1]
            if not uid_token.startswith("U"):
                continue
            uid = int(sequence[1][1:])
            if len(self.topk[uid]) >= self.K:
                continue
            recommended_token = sequence[-1]
            if not recommended_token.startswith("P"):
                continue
            recommended_item = int(recommended_token[1:])
            if recommended_item not in self.user_negatives[uid]:
                continue
            if recommended_item in self.topk[uid]:
                continue
            self.topk[uid].append(recommended_item)
            self.topk_sequences[uid].append(sequence)

    def reset_topks(self):
        del self.topk
        del self.topk_sequences
        self.topk = defaultdict(list)
        self.topk_sequences = defaultdict(list)
