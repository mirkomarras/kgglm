import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import BilinearModel

class DistMult(BilinearModel):
    """Implementation of DistMult model detailed in 2014 paper by Yang et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

    References
    ----------
    * Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng.
      `Embedding Entities and Relations for Learning and Inference in
      Knowledge Bases. <https://arxiv.org/abs/1412.6575>`_
      arXiv :1412.6575 [cs], December 2014.


    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_embeddings: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.

    """
    def __init__(self, num_entities, num_relations,margin, dim=100):
        super().__init__(dim,num_entities, num_relations)
        self.dim = dim # emb dim
        self.margin=margin
        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_emb = nn.Embedding(num_relations, self.dim)
        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)
        self.normalize_parameters()
        self.loss = nn.MarginRankingLoss(margin=self.margin)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2,dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_embeddings: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.

        """
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_emb.weight.data
    

    def forward(self, h, t, nh, nt, r,nr=None):
        """

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled relations.

        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.

        """
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

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`h^T \\cdot diag(r) \\cdot t`. See referenced paper for more
        details on the score. See torchkge.models.interfaces.Models for more
        details on the API.

        """
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)

        return (h * r * t).sum(dim=1)
