#https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/rotate.html#RotatE
#https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/base.html#KGEModel
#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/kge_fb15k_237.py
import torch
from torch import nn, empty
import torch.nn.functional as F
import math
from torchkge.models.interfaces import BilinearModel


class BaseRotatE(BilinearModel):
    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)
        self.emb_dim=emb_dim
        self.n_entities=n_entities
        self.n_relations=n_relations

        self.ent_emb_im = nn.Embedding(self.n_entities, self.emb_dim)
        self.ent_emb= nn.Embedding(self.n_entities, self.emb_dim)
        self.rel_emb = nn.Embedding(self.n_entities, self.emb_dim)

        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.ent_emb_im.weight)
        nn.init.uniform_(self.rel_emb.weight, 0, 2 * math.pi)
        
    def get_embeddings(self):
        """Return the embeddings of entities and matrices of relations.

        Returns
        -------
        ent_embeddings: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_mat: torch.Tensor, shape: (n_rel, emb_dim, emb_dim),
        dtype: torch.float
            Matrices of relations.

        """
        return self.re_ent_emb.weight.data, self.im_ent_emb.weight.data,self.re_rel_emb.weight.data, self.im_rel_emb.weight.data
    
    def forward(self, h, t, nh, nt, r,nr=None):

        pos = self.scoring_function(h,t, r)

        if nr is None:
            nr = r

        if nh.shape[0] > nr.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(nh.shape[0] / nr.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(nh,nt,nr.repeat(n_neg))
        else:
            neg = self.scoring_function(nh,nt,nr)

        return pos, neg


class RotatE(BaseRotatE):
    def scoring_function(self, h_idx, t_idx, r_idx):
        self.margin=9.0
        head_re = self.ent_emb(h_idx)
        head_im = self.ent_emb_im(h_idx)
        tail_re = self.ent_emb(t_idx)
        tail_im = self.ent_emb_im(t_idx)

        rel_theta = self.rel_emb(r_idx)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin - score

class BinaryCrossEntropyLoss(nn.Module):
    """This class implements :class:`torch.nn.Module` interface.

    """
    def __init__(self):
        super().__init__()

    def forward(self, positive_triplets, negative_triplets):
        """

        Parameters
        ----------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the true triplets as returned by the `forward` methods
            of the models.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scores of the negative triplets as returned by the `forward`
            methods of the models.
        Returns
        -------
        loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
            Loss of the form :math:`-\\eta \\cdot \\log(f(h,r,t)) +
            (1-\\eta) \\cdot \\log(1 - f(h,r,t))` where :math:`f(h,r,t)`
            is the score of the fact and :math:`\\eta` is either 1 or
            0 if the fact is true or false.
        """
        scores = torch.cat([positive_triplets, negative_triplets], dim=0)
        targets = torch.cat([torch.ones_like(positive_triplets), torch.zeros_like(negative_triplets)], dim=0)
        return F.binary_cross_entropy_with_logits(scores, targets)


