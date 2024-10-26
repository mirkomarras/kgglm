import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import BilinearModel


class RESCAL(BilinearModel):
    """Implementation of RESCAL model detailed in 2011 paper by Nickel et al..
    In the original paper, optimization is done using Alternating Least Squares
    (ALS). Here we use iterative gradient descent optimization. This class
    inherits from the :class:`torchkge.models.interfaces.BilinearModel`
    interface. It then has its attributes as well.

    References
    ----------
    * Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel.
      `A Three-way Model for Collective Learning on Multi-relational Data.
      <https://dl.acm.org/citation.cfm?id=3104584>`_
      In Proceedings of the 28th International Conference on Machine Learning,
      2011.


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
    rel_mat: torch.nn.Embedding, shape: (n_rel, emb_dim x emb_dim)
        Matrices of the relations, initialized with Xavier uniform
        distribution.

    """

    def __init__(self, dim, num_entities, num_relations, margin=1):
        super().__init__(dim, num_entities, num_relations)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin = margin
        self.ent_embeddings = nn.Embedding(self.num_entities, self.dim)
        self.rel_mat = nn.Embedding(self.num_relations, self.emb_dim * self.emb_dim)

        self.loss = nn.MarginRankingLoss(margin=self.margin)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_mat.weight.data)

        self.normalize_parameters()

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_embeddings.weight.data = F.normalize(
            self.ent_embeddings.weight.data, p=2, dim=1
        )

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
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_mat.weight.data.view(
            -1, self.emb_dim, self.emb_dim
        )

    def forward(self, h, t, nh, nt, r, nr=None):
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
        pos = self.scoring_function(h, t, r)

        if nr is None:
            nr = r

        if nh.shape[0] > nr.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(nh.shape[0] / nr.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(nh, nt, nr.repeat(n_neg))
        else:
            neg = self.scoring_function(nh, nt, nr)

        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`h^T \\cdot M_r \\cdot t`. See referenced paper for more details
        on the score. See torchkge.models.interfaces.Models for more details
        on the API.

        """
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)
        hr = torch.matmul(h.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t).sum(dim=1)
