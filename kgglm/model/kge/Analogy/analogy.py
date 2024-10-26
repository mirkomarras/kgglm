import torch
from torch import nn


class Analogy(nn.Module):
    """Implementation of ANALOGY model detailed in 2017 paper by Liu et al..
    According to their remark in the implementation details, the number of
    scalars on the diagonal of each relation-specific matrix is by default set
    to be half the embedding dimension.

    References
    ----------
    * Hanxiao Liu, Yuexin Wu, and Yiming Yang.
      `Analogical Inference for Multi-Relational Embeddings.
      <https://arxiv.org/abs/1705.02426>`_
      arXiv :1705.02426 [cs], May 2017.

    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    scalar_share: float
        Share of the diagonal elements of the relation-specific matrices to be
        scalars. By default it is set to half according to the original paper.

    Attributes
    ----------
    scalar_dim: int
        Number of diagonal elements of the relation-specific matrices to be
        scalars. By default it is set to half the embedding dimension according
        to the original paper.
    complex_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    sc_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Part of the entities embeddings associated to the scalar part of the
        relation specific matrices. Initialized with Xavier uniform
        distribution.
    re_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Real part of the entities complex embeddings. Initialized with Xavier
        uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    im_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Imaginary part of the entities complex embeddings. Initialized with
        Xavier uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    sc_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Part of the entities embeddings associated to the scalar part of the
        relation specific matrices. Initialized with Xavier uniform
        distribution.
    re_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Real part of the relations complex embeddings. Initialized with Xavier
        uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    im_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Imaginary part of the relations complex embeddings. Initialized with
        Xavier uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    """

    def __init__(self, emb_dim, n_entities, n_relations, scalar_share=0.5):
        super(Analogy, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.emb_dim = emb_dim

        self.scalar_dim = int(self.emb_dim * scalar_share)
        self.complex_dim = int((self.emb_dim - self.scalar_dim))

        self.sc_ent_emb = nn.Embedding(self.n_entities, self.scalar_dim)
        self.re_ent_emb = nn.Embedding(self.n_entities, self.complex_dim)
        self.im_ent_emb = nn.Embedding(self.n_entities, self.complex_dim)

        self.sc_rel_emb = nn.Embedding(self.n_relations, self.scalar_dim)
        self.re_rel_emb = nn.Embedding(self.n_relations, self.complex_dim)
        self.im_rel_emb = nn.Embedding(self.n_relations, self.complex_dim)

        nn.init.xavier_uniform_(self.sc_ent_emb.weight)
        nn.init.xavier_uniform_(self.re_ent_emb.weight)
        nn.init.xavier_uniform_(self.im_ent_emb.weight)
        nn.init.xavier_uniform_(self.sc_rel_emb.weight)
        nn.init.xavier_uniform_(self.re_rel_emb.weight)
        nn.init.xavier_uniform_(self.im_rel_emb.weight)

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        sc_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Scalar part of embeddings of entities.
        re_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Real part of embeddings of entities.
        im_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Imaginary part of embeddings of entities.
        sc_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Scalar part of embeddings of relations.
        re_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Real part of embeddings of relations.
        im_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Imaginary part of embeddings of relations.

        """
        return (
            self.sc_ent_emb.weight.data,
            self.re_ent_emb.weight.data,
            self.im_ent_emb.weight.data,
            self.sc_rel_emb.weight.data,
            self.re_rel_emb.weight.data,
            self.im_rel_emb.weight.data,
        )

    def forward(self, h, t, nh, nt, r, nr=None):
        """
        Parameters
        ----------
        h: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        t: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        r: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.
        nh: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        nt: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        nr: torch.Tensor, dtype: torch.long, shape: (batch_size)
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
        :math:`h_{sc}^T \\cdot diag(r_{sc}) \\cdot t_{sc} + \\Re(h_{compl}
        \\cdot diag(r_{compl} \\cdot t_{compl}))`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.
        """

        sc_h, re_h, im_h = (
            self.sc_ent_emb(h_idx),
            self.re_ent_emb(h_idx),
            self.im_ent_emb(h_idx),
        )
        sc_t, re_t, im_t = (
            self.sc_ent_emb(t_idx),
            self.re_ent_emb(t_idx),
            self.im_ent_emb(t_idx),
        )
        sc_r, re_r, im_r = (
            self.sc_rel_emb(r_idx),
            self.re_rel_emb(r_idx),
            self.im_rel_emb(r_idx),
        )

        return (sc_h * sc_r * sc_t).sum(dim=1) + (
            re_h * (re_r * re_t + im_r * im_t) + im_h * (re_r * im_t - im_r * re_t)
        ).sum(dim=1)

    def loss(self, pos, neg, pos_labels, neg_labels):
        # official paper : https://arxiv.org/pdf/1705.02426v2.pdf
        # https://ceur-ws.org/Vol-2377/paper_1.pdf <- Pointwise Logistic Loss | in the official paper they use the logistic loss (6.4.1)
        scores = torch.cat([pos, neg])
        pos_neg_labels = torch.cat([pos_labels, neg_labels])
        softplus = nn.Softplus()
        # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
        return torch.sum(softplus(-pos_neg_labels * scores))
