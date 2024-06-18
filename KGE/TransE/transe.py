"""

TRANS-E MODEL

"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import TranslationModel

class TransE(TranslationModel):
    """Implementation of TransE model detailed in 2013 paper by Bordes et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.


    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and
      Oksana Yakhnenko.
      `Translating Embeddings for Modeling Multi-relational Data.
      <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
      In Advances in Neural Information Processing Systems 26, pages 2787-795,
      2013.

    Parameters
    ----------
    emb_dim: int
        Dimension of the embedding.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    dissimilarity_type: str
        Either 'L1' or 'L2'.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.

    """
    def __init__(self, num_entities, num_relations, margin=1,dim=100):
        super().__init__(num_entities, num_relations, dissimilarity_type='L2')
        self.margin=margin
        self.dim = dim 
        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        
        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.normalize_parameters()
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

        self.loss = nn.MarginRankingLoss(margin=self.margin)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This method should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)

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
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data

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

    def scoring_function(self, h, t, r):
        """Compute the scoring function for the triplets given as argument:
        :math:`||h + r - t||_p^p` with p being the `dissimilarity type (either
        1 or 2)`. See referenced paper for more details
        on the score. See torchkge.models.interfaces.Models for more details
        on the API.

        """
        h = F.normalize(self.ent_embeddings(h), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t), p=2, dim=1)
        r = self.rel_embeddings(r)
        return -self.dissimilarity(h+r,t)

