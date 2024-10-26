import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.init import xavier_normal_


# https://github.com/TimDettmers/ConvE/blob/master/model.py#L79
class ConvE(nn.Module):
    def __init__(self, emb_dim, num_entities, num_relations, args):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, emb_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, emb_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = emb_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim)
        self.register_parameter("b", Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, emb_dim)

        self.loss = nn.BCELoss()

        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1_idx, _, r_idx):
        e1_embedded = self.emb_e(e1_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred

    #
    #
    # def forward(self, h, t, nh, nt, r, nr=None):
    #     """
    #
    #     Parameters
    #     ----------
    #     heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
    #         Integer keys of the current batch's heads
    #     tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
    #         Integer keys of the current batch's tails.
    #     relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
    #         Integer keys of the current batch's relations.
    #     negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
    #         Integer keys of the current batch's negatively sampled heads.
    #     negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
    #         Integer keys of the current batch's negatively sampled tails.ze)
    #     negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
    #         Integer keys of the current batch's negatively sampled relations.
    #
    #     Returns
    #     -------
    #     positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
    #         Scoring function evaluated on true triples.
    #     negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
    #         Scoring function evaluated on negatively sampled triples.
    #
    #     """
    #     pos = self.scoring_function(h, t, r)
    #
    #     if nr is None:
    #         nr = r
    #     # print(nh.shape,nr.shape)# 320 , 64
    #     if nh.shape[0] > nr.shape[0]:
    #         # in that case, several negative samples are sampled from each fact
    #         n_neg = int(nh.shape[0] / nr.shape[0])
    #         pos = pos.repeat(n_neg)
    #         neg = self.scoring_function(nh, nt, nr.repeat(n_neg))
    #     else:
    #         neg = self.scoring_function(nh, nt, nr)
    #
    #     return pos, neg


# class BinaryCrossEntropyLoss(nn.Module):
#     """This class implements :class:`torch.nn.Module` interface.
#
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, positive_triplets, negative_triplets, args):
#         """
#
#         Parameters
#         ----------
#         positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
#             Scores of the true triplets as returned by the `forward` methods
#             of the models.
#         negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
#             Scores of the negative triplets as returned by the `forward`
#             methods of the models.
#         Returns
#         -------
#         loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
#             Loss of the form :math:`-\\eta \\cdot \\log(f(h,r,t)) +
#             (1-\\eta) \\cdot \\log(1 - f(h,r,t))` where :math:`f(h,r,t)`
#             is the score of the fact and :math:`\\eta` is either 1 or
#             0 if the fact is true or false.
#         """
#         scores = torch.cat([positive_triplets, negative_triplets], dim=0)
#         targets = torch.cat([torch.ones_like(positive_triplets), torch.zeros_like(negative_triplets)], dim=0)
#         # Label Smoothing
#         if args.label_smoothing:
#             targets = ((1.0 - args.label_smoothing) * targets) + (1.0 / targets.size(1))
#         return F.binary_cross_entropy_with_logits(scores, targets)
