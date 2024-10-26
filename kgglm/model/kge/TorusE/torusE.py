from torch import nn
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import TranslationModel


class TorusE(TranslationModel):
    """Implementation of TorusE model detailed in 2018 paper by Ebisu and
    Ichise. This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Takuma Ebisu and Ryutaro Ichise
      `TorusE: Knowledge Graph Embedding on a Lie Group.
      <https://arxiv.org/abs/1711.05435>`_
      In Proceedings of the 32nd AAAI Conference on Artificial Intelligence
      (New Orleans, LA, USA, Feb. 2018), AAAI Press, pp. 1819-1826.

    Parameters
    ----------
    emb_dim: int
        Embedding dimension.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    dissimilarity_type: str
        One of 'torus_L1', 'torus_L2', 'torus_eL2'.

    Attributes
    ----------
    emb_dim: int
        Embedding dimension.
    ent_embeddings: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_embeddings: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.

    """

    def __init__(self, num_entities, num_relations, margin, dim=100):
        # One of 'torus_L1', 'torus_L2', 'torus_eL2'.
        super().__init__(num_entities, num_relations, dissimilarity_type="torus_L2")
        self.dim = dim
        self.margin = margin

        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)
        self.normalize_parameters()
        self.loss = nn.MarginRankingLoss(margin=margin)

    def normalize_parameters(self):
        """Project embeddings on torus."""
        self.ent_embeddings.weight.data.frac_()
        self.rel_embeddings.weight.data.frac_()

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_embeddings: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_embeddings: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.

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

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        See referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        h = self.ent_embeddings(h_idx)
        t = self.ent_embeddings(t_idx)
        r = self.rel_embeddings(r_idx)

        h.data.frac_()
        t.data.frac_()
        r.data.frac_()

        return -self.dissimilarity(h + r, t)
