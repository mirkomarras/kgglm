import torch
from torch import nn
import torch.nn.functional as F
from torchkge.models.interfaces import Model

class HolE(Model):
    """Implementation of HolE model detailed in 2015 paper by Nickel et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

    References
    ----------
    * Maximilian Nickel, Lorenzo Rosasco, and Tomaso Poggio.
      `Holographic Embeddings of Knowledge Graphs.
      <https://arxiv.org/abs/1510.04935>`_
      arXiv :1510.04935 [cs, stat], October 2015.

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
    ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.

    """

    def __init__(self, emb_dim, n_entities, n_relations,margin):
        super().__init__(n_entities, n_relations)
        self.n_ent=n_entities
        self.n_rel=n_relations
        self.emb_dim=emb_dim
        self.margin=margin

        self.ent_emb = nn.Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = nn.Embedding(self.n_rel, self.emb_dim)

        self.normalize_parameters()

        self.sigmoid = nn.Sigmoid()  # Check the original Paper
        self.MarginLoss = nn.MarginRankingLoss(margin=self.margin)

    def get_rolling_matrix(self,x):
        """Build a rolling matrix.

        Parameters
        ----------
        x: torch.Tensor, shape: (b_size, dim)

        Returns
        -------
        mat: torch.Tensor, shape: (b_size, dim, dim)
            Rolling matrix such that mat[i,j] = x[j - i mod(dim)]
        """
        b_size, dim = x.shape
        x = x.view(b_size, 1, dim)
        return torch.cat([x.roll(i, dims=2) for i in range(dim)], dim=1)
    
    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data, p=2,dim=1)
    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.

        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data
    
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
        :math:`h^T \\cdot M_r \\cdot t` where :math:`M_r` is the rolling matrix
        built from the relation embedding `r`. See referenced paper for more
        details on the score. See torchkge.models.interfaces.Models for more
        details on the API.

        """
        h = F.normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.get_rolling_matrix(self.rel_emb(r_idx))
        hr = torch.matmul(h.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t).sum(dim=1)

    def loss(self, positive_scores, negative_scores):
        return self.MarginLoss(self.sigmoid(positive_scores), self.sigmoid(negative_scores), torch.ones_like(positive_scores))
