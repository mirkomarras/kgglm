import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import TranslationModel


class TransH(TranslationModel):
    """Implementation of TransH model detailed in 2014 paper by Wang et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      `Knowledge Graph Embedding by Translating on Hyperplanes.
      <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531>`_
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.

    Parameters
    ----------
    emb_dim: int
        Dimension of the embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.
    norm_vect: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Normal vectors associated to each relation and used to compute the
        relation-specific hyperplanes entities are projected on. See paper for
        more details. Initialized with Xavier uniform distribution and then
        normalized.

    """

    def __init__(self, num_entities, num_relations, margin, dim=100):
        super(TransH, self).__init__(
            num_entities, num_relations, dissimilarity_type="L2"
        )
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin = margin
        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        self.norm_vect = nn.Embedding(num_relations, self.dim)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.normalize_parameters()

        self.loss = self.loss = nn.MarginRankingLoss(margin=self.margin)

        self.projected_entities = nn.Parameter(
            torch.empty(size=(num_relations, num_entities, self.dim)),
            requires_grad=False,
        )

    def normalize_parameters(self):
        """Normalize the entity embeddings and relations normal vectors, as
        explained in original paper. This methods should be called at the end
        of each training epoch and at the end of training as well.

        """
        self.ent_embeddings.weight.data = F.normalize(
            self.ent_embeddings.weight.data, p=2, dim=1
        )
        self.rel_embeddings.weight.data = F.normalize(
            self.rel_embeddings.weight.data, p=2, dim=1
        )
        self.norm_vect.weight.data = F.normalize(self.norm_vect.weight.data, p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with relation
        normal vectors.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.
        norm_vect: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Normal vectors defining relation-specific hyperplanes.
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

    def project(self, ent, norm_vect):
        return ent - (ent * norm_vect).sum(dim=1).view(-1, 1) * norm_vect

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`||p_r(h) + r - p_r(t)||_2^2`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = self.rel_embeddings(r_idx)
        norm_vect = F.normalize(self.norm_vect(r_idx), p=2, dim=1)
        return -self.dissimilarity(
            self.project(h, norm_vect=norm_vect) + r,
            self.project(t, norm_vect=norm_vect),
        )
