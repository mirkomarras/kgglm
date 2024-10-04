'''
Created on Dec 03, 2023
Pytorch Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from typing import Dict, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn


class KGAT(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(KGAT, self).__init__()
        self._parse_args(data_config, pretrain_data, args)
        self.device = args.device
        self._build_weights()
        # Setting the aggregation function based on the alg_type
        method_mapper = {
            "bi": self.create_bi_interaction_embed,
            "kgat": self.create_bi_interaction_embed,
            #"gcn": self._gcn_method,
            #"graphsage": self._graphsage_method
        }

        if self.alg_type in method_mapper:
            self.aggregation_fun = lambda: method_mapper[self.alg_type]
        else:
            raise NotImplementedError
        # Define the two sets of parameters for each optimizer
        # Correctly gather parameters for the recommendation optimizer
        rec_params = []
        rec_params.extend(self.user_embed.parameters())
        rec_params.extend(self.entity_embed.parameters())
        #rec_params.extend(self.gc_layers.parameters())
        #rec_params.extend(self.bi_layers.parameters())
        #rec_params.extend(self.mlp_layers.parameters())
        # Include other parameters related to 'rec' if necessary

        # Correctly gather parameters for the KGE optimizer
        kge_params = []
        kge_params.extend(self.relation_embed.parameters())
        kge_params.append(self.trans_W)

        # Initialize the two optimizers
        self.recommendation_opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.kge_opt = torch.optim.Adam(self.parameters(), lr=args.lr)
        self._statistics_params()

    def _define_layer(self, in_size, out_size, name):
        setattr(self, f'W_{name}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_size, out_size))))
        setattr(self, f'b_{name}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, out_size))))

    def _build_weights(self):
        if self.pretrain_data is None:
            self.user_embed = torch.nn.Embedding(self.n_users, self.emb_dim).to(self.device)
            self.entity_embed = torch.nn.Embedding(self.n_entities, self.emb_dim).to(self.device)
            torch.nn.init.xavier_uniform_(self.user_embed.weight.data)
            torch.nn.init.xavier_uniform_(self.entity_embed.weight.data)
            print('using xavier initialization, for user and entity embed')
        else:
            self.user_embed = nn.Parameter(
                torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32)).to(self.device)

            item_embed = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32).to(self.device)
            other_embed = torch.nn.init.xavier_uniform_(torch.empty(self.n_entities - self.n_items, self.emb_dim)).to(self.device)

            self.entity_embed = nn.Parameter(torch.cat([item_embed, other_embed], dim=0)).to(self.device)
            print('using pretrained initialization')

        #self.relation_embed = nn.Parameter(
        #    torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.kge_dim)))
        self.relation_embed = torch.nn.Embedding(self.n_relations, self.kge_dim).to(self.device)
        torch.nn.init.xavier_uniform_(self.relation_embed.weight.data)

        self.trans_W = nn.Parameter(torch.empty(self.n_relations, self.emb_dim, self.kge_dim))
        nn.init.xavier_uniform_(self.trans_W)
        #self.trans_W = nn.Embedding(self.n_relations, self.emb_dim * self.kge_dim)
        #nn.init.xavier_uniform_(self.trans_W.weight)

        self.weight_size_list = [self.emb_dim] + self.weight_size

        self.gc_layers, self.bi_layers, self.mlp_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for k in range(self.n_layers):
            # Graph Convolutional Layers
            W_gc = nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1])))
            b_gc = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1])))
            self.gc_layers.append(nn.Linear(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.register_parameter(f'W_gc_{k}', W_gc)
            self.register_parameter(f'b_gc_{k}', b_gc)

            # Bilinear Layers
            W_bi = nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1])))
            b_bi = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1])))
            self.bi_layers.append(nn.Linear(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.register_parameter(f'W_bi_{k}', W_bi)
            self.register_parameter(f'b_bi_{k}', b_bi)

            # MLP Layers
            W_mlp = nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(2 * self.weight_size_list[k], self.weight_size_list[k + 1])))
            b_mlp = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1])))
            self.mlp_layers.append(nn.Linear(2 * self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.register_parameter(f'W_mlp_{k}', W_mlp)
            self.register_parameter(f'b_mlp_{k}', b_mlp)
        self.A_values = torch.FloatTensor(len(self.all_h_list)).to(self.device)


    def _initialize_tensor(self, value: float = 0.0) -> torch.Tensor:
        """Initialize a tensor with the given value."""
        return torch.tensor(value).float().to(self.device)

    def train_step(self, batch_data, mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one training step based on the given mode ('rec' or 'kge').

        Args:
            batch_data: Data required for the training step.
            mode (str): Specifies the mode of operation, either 'rec' for recommendation or 'kge' for knowledge graph embedding.

        Returns:
            Tuple of batch loss, base loss, kge loss, and regularization loss.
        """
        base_loss = self._initialize_tensor()
        kge_loss = self._initialize_tensor()
        reg_loss = self._initialize_tensor()
        batch_loss = self._initialize_tensor()

        if mode == 'rec':
            u_e, pos_i_e, neg_i_e = self.forward(batch_data, mode)
            base_loss, reg_loss, recommendation_loss = self._bpr_loss(u_e, pos_i_e, neg_i_e)
            batch_loss = recommendation_loss
            opt = self.recommendation_opt

        elif mode == 'kge':
            h_e, r_e, pos_t_e, neg_t_e = self.forward(batch_data, mode)
            kge_loss, reg_loss = self._kge_loss(h_e, r_e, pos_t_e, neg_t_e)
            batch_loss = kge_loss
            opt = self.kge_opt
        else:
            raise ValueError("Invalid mode")

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        return batch_loss, base_loss, kge_loss, reg_loss

    def forward(self, batch_data: Dict[str, torch.Tensor], mode: str) -> Union[torch.Tensor, None]:
        """
        Forward pass based on the given mode.

        Args:
            batch_data: Data required for the forward pass.
            mode (str): Specifies the mode of operation.

        Returns:
            Result of the forward pass or None.
        """
        if mode == 'rec':
            #users, pos_items, neg_items = batch_data
            return self._forward_phase_I(batch_data['users'], batch_data['pos_items'], batch_data['neg_items'])
        elif mode == 'kge':
            heads, relations, pos_tails, neg_tails = [torch.IntTensor(x).to(self.device) for x in batch_data]
            return self._forward_phase_II(heads, relations, pos_tails, neg_tails)
        elif mode == 'eval':
            u_e, pos_i_e, _ = self._forward_phase_I(batch_data['users'], batch_data['pos_items'])
            return torch.matmul(u_e, pos_i_e.t())  # Transpose pos_i_e for matrix multiplication

        else:
            raise ValueError("Invalid mode")

    """
    *********************************************************
    Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
    Different Convolutional Layers:
        1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
        2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
        3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
    """
    def _forward_phase_I(self, users, pos_items, neg_items=None):
        ua_embeddings, ea_embeddings = self.create_bi_interaction_embed()
        u_e, pos_i_e, neg_i_e = ua_embeddings[users], ea_embeddings[pos_items], ea_embeddings[neg_items]
        return u_e, pos_i_e, neg_i_e
    """
    *********************************************************
    Compute Knowledge Graph Embeddings via TransR.
    """

    def _forward_phase_II(self, h, r, pos_t, neg_t):
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_inference(h, r, pos_t, neg_t)
        self.A_kg_score = self._generate_transE_score(h=h, t=pos_t, r=r)
        self.A_out = self._create_attentive_A_out()
        return h_e, r_e, pos_t_e, neg_t_e

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        """
                Retrieve the knowledge graph embeddings for given entities and relations.

                Given tensors of head entities (h), relations (r), positive tail entities (pos_t),
                and negative tail entities (neg_t), this function returns their respective embeddings.

                Args:
                    h (torch.Tensor): Tensor of head entities.
                    r (torch.Tensor): Tensor of relations.
                    pos_t (torch.Tensor): Tensor of positive tail entities.
                    neg_t (torch.Tensor): Tensor of negative tail entities.

                Returns:
                    Tuple[torch.Tensor, ...]: A tuple containing transformed embeddings for head entities,
                                              relation embeddings, positive tail entities, and negative tail entities.
                """
        embeddings = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        embeddings = embeddings.unsqueeze(1)

        # Embedding lookup for head & tail entities
        h_e = embeddings[h]
        pos_t_e = embeddings[pos_t]
        neg_t_e = embeddings[neg_t]

        # Relation embeddings
        r_e = self.relation_embed(r)

        # Relation transform weights
        trans_M = self.trans_W[r]

        # Perform matrix multiplication and reshape
        h_e = torch.matmul(h_e, trans_M).view(-1, self.kge_dim)
        pos_t_e = torch.matmul(pos_t_e, trans_M).view(-1, self.kge_dim)
        neg_t_e = torch.matmul(neg_t_e, trans_M).view(-1, self.kge_dim)

        # L2 normalization can be added here if needed

        return h_e, r_e, pos_t_e, neg_t_e


    """
    Optimize Recommendation (CF) Part via BPR Loss.
    """

    def _bpr_loss(self, u_e, pos_i_e, neg_i_e):
        pos_scores = torch.sum(u_e * pos_i_e, dim=1)
        neg_scores = torch.sum(u_e * neg_i_e, dim=1)
        bpr_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        # Regularization
        reg_loss = (u_e.norm(p=2).pow(2) +
                    pos_i_e.norm(p=2).pow(2) +
                    neg_i_e.norm(p=2).pow(2))
        reg_loss = (reg_loss / self.batch_size) * self.regs[0]

        loss = bpr_loss + reg_loss
        return bpr_loss, reg_loss, loss

    def _kge_loss(self, h_e, r_e, pos_t_e, neg_t_e):
        pos_kg_score = torch.sum((h_e + r_e - pos_t_e).pow(2), dim=1, keepdim=True)
        neg_kg_score = torch.sum((h_e + r_e - neg_t_e).pow(2), dim=1, keepdim=True)
        # Using the softplus as BPR loss to avoid the nan error.
        kg_loss = F.softplus(-(neg_kg_score - pos_kg_score)).mean()
        kg_reg_loss = h_e.norm(p=2).pow(2) + r_e.norm(p=2).pow(2) + \
                      pos_t_e.norm(p=2).pow(2) + neg_t_e.norm(p=2).pow(2)
        kg_reg_loss = kg_reg_loss / self.batch_size_kg
        reg_loss = self.regs[1] * kg_reg_loss
        kge_loss = kg_loss + reg_loss
        return kge_loss, reg_loss

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape)

    def _create_attentive_A_out(self):
        indices = torch.LongTensor(np.vstack([self.all_h_list, self.all_t_list]).transpose()).to(self.device)
        A = torch.sparse.softmax(torch.sparse_coo_tensor(indices.t(), self.A_values, self.A_in.shape), dim=1)
        return A


    def update_attentive_A(self):
        """
        Update the adjacency matrix A using attentive TransE scores.

        This function computes the attentive adjacency matrix by splitting the data into
        batches, computing the TransE scores for each batch, and then constructing a new
        adjacency matrix using the scores.
        """
        fold_len = len(self.all_h_list) // self.n_fold
        kg_scores = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = len(self.all_h_list) if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len

            h_batch = torch.tensor(self.all_h_list[start:end], dtype=torch.long).to(self.device)
            r_batch = torch.tensor(self.all_r_list[start:end], dtype=torch.long).to(self.device)
            pos_t_batch = torch.tensor(self.all_t_list[start:end], dtype=torch.long).to(self.device)

            A_kg_score = self._generate_transE_score(h_batch, pos_t_batch, r_batch)
            kg_scores.append(A_kg_score.detach().cpu().numpy())

        kg_scores = np.concatenate(kg_scores)

        # Create the attentive adjacency matrix
        indices = np.vstack([self.all_h_list, self.all_t_list]).transpose()
        A_in_coo = sp.coo_matrix((kg_scores, (indices[:, 0], indices[:, 1])),
                                 shape=(self.n_users + self.n_entities, self.n_users + self.n_entities))
        self.A_in = A_in_coo.tocsr()

        if self.alg_type in ['org', 'gcn']:
            # Adding diagonal elements
            self.A_in.setdiag(np.ones(self.n_users + self.n_entities))

    def _generate_transE_score(self, h: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # Concatenate user and entity embeddings
        embeddings = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        embeddings = embeddings.unsqueeze(1)

        # Embedding lookup for head and tail entities
        h_e = embeddings[h]
        t_e = embeddings[t]

        # Relation embeddings
        r_e = self.relation_embed(r)

        # Relation transform weights
        trans_M = self.trans_W[r]

        # Perform matrix multiplication and reshape
        h_e = torch.matmul(h_e, trans_M).view(-1, self.kge_dim)
        t_e = torch.matmul(t_e, trans_M).view(-1, self.kge_dim)

        # L2 normalization can be added here if needed

        # Calculate TransE score
        kg_score = torch.sum(t_e * torch.tanh(h_e + r_e), dim=1)

        return kg_score

    def _statistics_params(self):
        # number of params
        total_parameters = sum(p.numel() for p in self.parameters())
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def create_bi_interaction_embed(self):
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(self.A_in)

        ego_embeddings = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            temp_embed = [torch.sparse.mm(A_fold.to(self.device), ego_embeddings) for A_fold in A_fold_hat]

            # Sum messages of neighbors.
            side_embeddings = torch.cat(temp_embed, 0)

            add_embeddings = ego_embeddings + side_embeddings

            # Transformed sum messages of neighbors.
            W_gc, b_gc = getattr(self, f'W_gc_{k}'), getattr(self, f'b_gc_{k}')
            sum_embeddings = F.leaky_relu(torch.mm(add_embeddings, W_gc) + b_gc)

            # Bi messages of neighbors.
            bi_embeddings = ego_embeddings * side_embeddings
            # Transformed bi messages of neighbors.
            W_bi, b_bi = getattr(self, f'W_bi_{k}'), getattr(self, f'b_bi_{k}')
            bi_embeddings = F.leaky_relu(torch.mm(bi_embeddings, W_bi) + b_bi)

            ego_embeddings = bi_embeddings + sum_embeddings

            # Message dropout.
            ego_embeddings = F.dropout(ego_embeddings, self.mess_dropout[k])

            # Normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)
        ua_embeddings, ea_embeddings = torch.split(all_embeddings, [self.n_users, self.n_entities], 0)

        return ua_embeddings, ea_embeddings


    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'kgat'
        self.device = args.device
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100

        # initialize the attentive matrix A for phase I.
        self.A_in = data_config['A_in']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)
        self.mess_dropout = eval(args.mess_dropout)
        self.regs = eval(args.regs)
        self.verbose = args.verbose




