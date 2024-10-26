import torch
import torch.nn.functional as F
from torch import nn


class ComplEx(nn.Module):
    """Implementation of ComplEx model detailed in 2016 paper by Trouillon et
    al.. This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

    References
    ----------
    * Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and
      Guillaume Bouchard.
      `Complex Embeddings for Simple Link Prediction.
      <https://arxiv.org/abs/1606.06357>`_
      arXiv :1606.06357 [cs, stat], June 2016.

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
    re_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Real part of the entities complex embeddings. Initialized with Xavier
        uniform distribution.
    im_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Imaginary part of the entities complex embeddings. Initialized with
        Xavier uniform distribution.
    re_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Real part of the relations complex embeddings. Initialized with Xavier
        uniform distribution.
    im_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Imaginary part of the relations complex embeddings. Initialized with
        Xavier uniform distribution.
    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super(ComplEx, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.emb_dim = emb_dim

        self.re_ent_emb = nn.Embedding(self.n_entities, self.emb_dim)
        self.im_ent_emb = nn.Embedding(self.n_entities, self.emb_dim)
        self.re_rel_emb = nn.Embedding(self.n_relations, self.emb_dim)
        self.im_rel_emb = nn.Embedding(self.n_relations, self.emb_dim)

        # from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/complex.html#ComplEx
        nn.init.xavier_uniform_(self.re_ent_emb.weight)
        nn.init.xavier_uniform_(self.im_ent_emb.weight)
        nn.init.xavier_uniform_(self.re_rel_emb.weight)
        nn.init.xavier_uniform_(self.im_rel_emb.weight)

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        re_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Real part of embeddings of entities.
        im_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Imaginary part of embeddings of entities.
        re_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Real part of embeddings of relations.
        im_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Imaginary part of embeddings of relations.

        """
        return (
            self.re_ent_emb.weight.data,
            self.im_ent_emb.weight.data,
            self.re_rel_emb.weight.data,
            self.im_rel_emb.weight.data,
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

    # from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/complex.html#ComplEx
    def triple_dot(self, x, y, z):
        return (x * y * z).sum(dim=-1)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the real part of the Hermitian product
        :math:`\\Re(h^T \\cdot diag(r) \\cdot \\bar{t})` for each sample of
        the batch. See referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        # from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/complex.html#ComplEx
        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)
        return (
            self.triple_dot(re_h, re_r, re_t)
            + self.triple_dot(im_h, re_r, im_t)
            + self.triple_dot(re_h, im_r, im_t)
            - self.triple_dot(im_h, im_r, im_t)
        )

    # from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/complex.html#ComplEx
    def loss(self, pos_score, neg_score):
        # https://github.com/pyg-team/pytorch_geometric/blob/2947af7733d87315a9218a0bfabbfda10b595f0e/torch_geometric/nn/kge/complex.py
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)
