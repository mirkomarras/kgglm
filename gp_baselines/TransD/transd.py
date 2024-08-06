import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import TranslationModel

class TransD(TranslationModel):
    """Implementation of TransD model detailed in 2015 paper by Ji et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao.
      `Knowledge Graph Embedding via Dynamic Mapping Matrix.
      <https://aclweb.org/anthology/papers/P/P15/P15-1067/>`_
      In Proceedings of the 53rd Annual Meeting of the Association for
      Computational Linguistics and the 7th International Joint Conference on
      Natural Language Processing (Volume 1: Long Papers) pages 687–696,
      Beijing, China, July 2015. Association for Computational Linguistics.

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
        Dimension of the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.
    ent_proj_vect: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Entity-specific vector used to build projection matrices. See paper for
        more details. Initialized with Xavier uniform distribution and then
        normalized.
    rel_proj_vect: `torch..nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Relation-specific vector used to build projection matrices. See paper
        for more details. Initialized with Xavier uniform distribution and then
        normalized.
    projected_entities: `torch.nn.Parameter`, \
        shape: (n_rel, n_ent, rel_emb_dim)
        Contains the projection of each entity in each relation-specific
        sub-space.
    evaluated_projections: bool
        Indicates whether `projected_entities` has been computed. This should
        be set to true every time a backward pass is done in train mode.

    """
    def __init__(self, num_entities, num_relations,margin, dim=100):
        super().__init__(num_entities, num_relations, dissimilarity_type='L2')
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin=margin

        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        self.ent_vect = nn.Embedding(num_entities, self.dim)
        self.rel_vect = nn.Embedding(num_relations, self.dim)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.loss = nn.MarginRankingLoss(margin=self.margin)

        self.normalize_parameters()
        self.projected_entities = nn.Parameter(torch.empty(size=(num_relations, num_entities, self.dim)), requires_grad=False)

    def normalize_parameters(self):
        """Normalize the entity embeddings and relations normal vectors, as
        explained in original paper. This methods should be called at the end
        of each training epoch and at the end of training as well.

        """
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_vect.weight.data = F.normalize(self.ent_vect.weight.data, p=2, dim=1)
        self.rel_vect.weight.data = F.normalize(self.rel_vect.weight.data, p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with their
        projection vectors.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, ent_emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, rel_emb_dim), dtype: torch.float
            Embeddings of relations.
        ent_proj_vect: torch.Tensor, shape: (n_ent, ent_emb_dim),
        dtype: torch.float
            Entity projection vectors.
        rel_proj_vect: torch.Tensor, shape: (n_ent, rel_emb_dim),
        dtype: torch.float
            Relation projection vectors.

        """
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data
    
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

    def project(self, ent, ent_vect, rel_vect):
        """We note that :math:`p_r(e)_i = e^p^Te \\times r^p_i + e_i` which is
        more efficient to compute than the matrix formulation in the original
        paper.

        """
        proj_e = (rel_vect * (ent * ent_vect).sum(dim=1).view(ent.shape[0], 1))
        return proj_e + ent[:, :self.dim]

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`||p_r(h) + r - p_r(t)||_2^2`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = F.normalize(self.rel_embeddings(r_idx), p=2, dim=1)

        h_proj = F.normalize(self.ent_vect(h_idx), p=2, dim=1)
        t_proj = F.normalize(self.ent_vect(t_idx), p=2, dim=1)
        r_proj = F.normalize(self.rel_vect(r_idx), p=2, dim=1)

        return -self.dissimilarity(self.project(h, h_proj, r_proj) + r, self.project(t, t_proj, r_proj))