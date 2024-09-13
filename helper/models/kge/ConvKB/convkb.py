import torch
from torch import nn
from torch import cuda
from torch.nn.init import xavier_uniform_
from torchkge.models.interfaces import Model

class ConvKB(Model):
    """Implementation of ConvKB model detailed in 2018 paper by Nguyen et al..
    This class inherits from the :class:`torchkge.models.interfaces.Model`
    interface. It then has its attributes as well.


    References
    ----------
    * Nguyen, D. Q., Nguyen, T. D., Nguyen, D. Q., and Phung, D.
      `A Novel Embed- ding Model for Knowledge Base Completion Based on
      Convolutional Neural Network.
      <https://arxiv.org/abs/1712.02121>`_
      In Proceedings of the 2018 Conference of the North American Chapter of
      the Association for Computational Linguistics: Human Language
      Technologies (2018), vol. 2, pp. 327-333.

    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_filters: int
        Number of filters used for convolution.
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
    def __init__(self, num_entities, num_relations,device,margin, emb_dim=100, n_filters=64,dropout=0.5):
        super().__init__(num_entities, num_relations)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.n_filters = n_filters
        self.device=device
        self.dropout=dropout
        self.margin=margin

        self.ent_embeddings = nn.Embedding(self.num_entities, self.emb_dim)
        self.rel_embeddings = nn.Embedding(self.num_relations, self.emb_dim)
        
        #check official code: https://github.com/daiquocnguyen/ConvKB/blob/master/ConvKB_pytorch/ConvKB_1D.py
        self.conv1_bn = nn.BatchNorm1d(3)
        self.conv_layer = nn.Conv1d(3, self.n_filters, 1,stride=1)  # kernel size x 3
        self.conv2_bn = nn.BatchNorm1d(self.n_filters)
        self.dropout = nn.Dropout(dropout)
        self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.fc_layer = nn.Linear(self.emb_dim * self.n_filters, 2, bias=False)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)
        xavier_uniform_(self.fc_layer.weight.data)
        xavier_uniform_(self.conv_layer.weight.data)

        self.convlayer = nn.Sequential(self.conv1_bn,self.conv_layer,self.conv2_bn,self.dropout,self.non_linearity)
        self.output = nn.Sequential(self.fc_layer, nn.Softmax(dim=1))

        self.loss = nn.MarginRankingLoss(margin=self.margin)

        


    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.
        """
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

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        by applying convolutions to the concatenation of the embeddings. See
        referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.
        
        Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's heads
        t_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's tails.
        r_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype: torch.float, shape: (b_size)
            Score of each triplet.
        """

        b_size = h_idx.shape[0]
        h = self.ent_embeddings(h_idx).view(b_size, 1, -1).to(self.device)
        t = self.ent_embeddings(t_idx).view(b_size, 1, -1).to(self.device)
    
        r = self.rel_embeddings(r_idx).view(b_size, 1, -1).to(self.device)

        concat = torch.cat((h, r, t), dim=1).to(self.device)

        return self.output(self.convlayer(concat).reshape(b_size, -1))[:, 1]
