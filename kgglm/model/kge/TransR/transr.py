import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import TranslationModel


class TransR(TranslationModel):
    """Implementation of TransR model detailed in 2015 paper by Lin et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu.
      `Learning Entity and Relation Embeddings for Knowledge Graph Completion.
      <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523>`_
      In Twenty-Ninth AAAI Conference on Artificial Intelligence, February 2015

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension nof the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.
    proj_mat: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim x ent_emb_dim)
        Relation-specific projection matrices. See paper for more details.
    projected_entities: `torch.nn.Parameter`, \
        shape: (n_rel, n_ent, rel_emb_dim)
        Contains the projection of each entity in each relation-specific
        sub-space.
    evaluated_projections: bool
        Indicates whether `projected_entities` has been computed. This should
        be set to true every time a backward pass is done in train mode.
    """

    def __init__(self, num_entities, num_relations, margin, dim=100):
        super(TransR, self).__init__(
            num_entities, num_relations, dissimilarity_type="L2"
        )
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin = margin
        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        self.proj_mat = nn.Embedding(num_relations, self.dim * self.dim)

        self.loss = nn.MarginRankingLoss(margin=margin)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.normalize_parameters()

        self.projected_entities = nn.Parameter(
            torch.empty(size=(num_relations, num_entities, self.dim)),
            requires_grad=False,
        )

    def normalize_parameters(self):
        """Normalize the entity and relation embeddings, as explained in
        original paper. This methods should be called at the end of each
        training epoch and at the end of training as well.

        """
        self.ent_embeddings.weight.data = F.normalize(
            self.ent_embeddings.weight.data, p=2, dim=1
        )
        self.rel_embeddings.weight.data = F.normalize(
            self.rel_embeddings.weight.data, p=2, dim=1
        )

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with their
        projection matrices.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, ent_emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, rel_emb_dim), dtype: torch.float
            Embeddings of relations.
        proj_mat: torch.Tensor, shape: (n_rel, rel_emb_dim, ent_emb_dim),
        dtype: torch.float
            Relation-specific projection matrices.
        """
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data

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

    def project(self, ent, proj_mat):
        proj_e = torch.matmul(proj_mat, ent.view(-1, self.dim, 1))
        return proj_e.view(-1, self.dim)

    def scoring_function(self, h_idx, t_idx, r_idx):
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = self.rel_embeddings(r_idx)

        proj_mat = self.proj_mat(r_idx).view(h_idx.shape[0], self.dim, self.dim)
        return -self.dissimilarity(
            self.project(h, proj_mat=proj_mat) + r, self.project(t, proj_mat=proj_mat)
        )


# class MarginLoss(nn.Module):
#     """Margin loss as it was defined in `TransE paper
#     <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
#     by Bordes et al. in 2013. This class implements :class:`torch.nn.Module`
#     interface.
#
#     """
#     def __init__(self, margin):
#         super().__init__()
#         self.loss = nn.MarginRankingLoss(margin=margin, reduction='sum')
#
#     def forward(self, positive_scores, negative_scores):
#         """
#         Parameters
#         ----------
#         positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
#             Scores of the true triplets as returned by the `forward` methods of
#             the models.
#         negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
#             Scores of the negative triplets as returned by the `forward`
#             methods of the models.
#
#         Returns
#         -------
#         loss: torch.Tensor, shape: (n_facts, dim), dtype: torch.float
#             Loss of the form
#             :math:`\\max\\{0, \\gamma - f(h,r,t) + f(h',r',t')\\}` where
#             :math:`\\gamma` is the margin (defined at initialization),
#             :math:`f(h,r,t)` is the score of a true fact and
#             :math:`f(h',r',t')` is the score of the associated negative fact.
#         """
#         return self.loss(positive_scores, negative_scores, target=torch.ones_like(positive_scores))
