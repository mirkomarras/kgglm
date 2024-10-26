import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class CKE(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(CKE, self).__init__()
        self.device = args.device
        self._parse_args(data_config, pretrain_data, args)
        self._build_model()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = "cke"
        self.pretrain_data = pretrain_data

        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_entities = data_config["n_entities"]
        self.n_relations = data_config["n_relations"]

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.kge_dim = args.kge_size
        self.regs = eval(args.regs)
        self.verbose = args.verbose

    def _build_model(self):
        if self.pretrain_data is None:
            self.user_embed = nn.Embedding(self.n_users, self.emb_dim)
            nn.init.xavier_uniform_(self.user_embed.weight.data)
            print("self.n_items -> ", self.n_items)
            self.item_embed = nn.Embedding(self.n_items, self.emb_dim)
            nn.init.xavier_uniform_(self.item_embed.weight.data)
            print("using xavier initialization")
        else:
            self.user_embed = nn.Embedding.from_pretrained(
                torch.Tensor(self.pretrain_data["user_embed"]), freeze=False
            )
            self.item_embed = nn.Embedding.from_pretrained(
                torch.Tensor(self.pretrain_data["item_embed"]), freeze=False
            )
            print("using pretrained initialization")

        self.kg_entity_embed = nn.Embedding(self.n_entities, self.emb_dim)
        nn.init.xavier_uniform_(self.kg_entity_embed.weight.data)
        self.kg_relation_embed = nn.Embedding(self.n_relations, self.kge_dim)
        nn.init.xavier_uniform_(self.kg_relation_embed.weight.data)
        self.trans_W = nn.Parameter(
            torch.Tensor(self.n_relations, self.emb_dim, self.kge_dim)
        )
        nn.init.xavier_normal_(self.trans_W)

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        # Embedding lookups
        h_e = self.kg_entity_embed(h)
        pos_t_e = self.kg_entity_embed(pos_t)
        neg_t_e = self.kg_entity_embed(neg_t)

        # Embedding lookup for relations
        r_e = self.kg_relation_embed(r)

        # Relation transform weights
        trans_M = self.trans_W[r]

        # Perform the transformation
        h_e = torch.matmul(h_e.unsqueeze(1), trans_M).squeeze(1)
        pos_t_e = torch.matmul(pos_t_e.unsqueeze(1), trans_M).squeeze(1)
        neg_t_e = torch.matmul(neg_t_e.unsqueeze(1), trans_M).squeeze(1)

        # L2 normalization
        h_e = F.normalize(h_e, p=2, dim=1)
        r_e = F.normalize(r_e, p=2, dim=1)
        pos_t_e = F.normalize(pos_t_e, p=2, dim=1)
        neg_t_e = F.normalize(neg_t_e, p=2, dim=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _cf_forward(self, u, pos_i, neg_i):
        u_e = self.user_embed(u)

        # Embedding lookup for positive and negative items
        pos_i_e = self.item_embed(pos_i)
        neg_i_e = self.item_embed(neg_i)

        # Embedding lookup for positive and negative items in KG embeddings
        pos_i_kg_e = self.kg_entity_embed(pos_i).view(-1, self.emb_dim)
        neg_i_kg_e = self.kg_entity_embed(neg_i).view(-1, self.emb_dim)

        # Combine item embeddings with their corresponding KG embeddings
        pos_i_combined = pos_i_e + pos_i_kg_e
        neg_i_combined = neg_i_e + neg_i_kg_e

        return u_e, pos_i_combined, neg_i_combined

    def _get_cf_inference(self, u, pos_i):
        u_e = self.user_embed(u)
        pos_i_e = self.item_embed(pos_i)

        # Embedding lookup for positive and negative items in KG embeddings
        pos_i_kg_e = self.kg_entity_embed(pos_i).view(-1, self.emb_dim)

        # Combine item embeddings with their corresponding KG embeddings
        pos_i_combined = pos_i_e + pos_i_kg_e

        return u_e, pos_i_combined

    def train_step(self, batch_data):
        u_e, pos_i_e, neg_i_e, h_e, r_e, pos_t_e, neg_t_e, batch_predictions = self(
            batch_data
        )
        self.optimizer.zero_grad()

        # Compute loss
        total_loss, base_loss, kge_loss, reg_loss = self._build_loss(
            u_e, pos_i_e, neg_i_e, h_e, r_e, pos_t_e, neg_t_e
        )

        # Backward pass and optimize
        total_loss.backward()
        self.optimizer.step()
        return total_loss, base_loss, kge_loss, reg_loss

    def forward(self, batch_data, mode="train"):
        if mode == "train":
            # Get CF and KG inferences
            u, pos_i, neg_i = (
                torch.IntTensor(batch_data["users"]).to(self.device),
                torch.IntTensor(batch_data["pos_items"]).to(self.device),
                torch.IntTensor(batch_data["neg_items"]).to(self.device),
            )
            u_e, pos_i_e, neg_i_e = self._cf_forward(u, pos_i, neg_i)
            h, r, pos_t, neg_t = (
                torch.IntTensor(batch_data["heads"]).to(self.device),
                torch.IntTensor(batch_data["relations"]).to(self.device),
                torch.IntTensor(batch_data["pos_tails"]).to(self.device),
                torch.IntTensor(batch_data["neg_tails"]).to(self.device),
            )
            h_e, r_e, pos_t_e, neg_t_e = self._get_kg_inference(h, r, pos_t, neg_t)
            batch_predictions = torch.matmul(u_e, pos_i_e.t())

            return u_e, pos_i_e, neg_i_e, h_e, r_e, pos_t_e, neg_t_e, batch_predictions

        elif mode == "eval":
            # embeddings = torch.cat([self.user_embed.weight, self.kg_entity_embed.weight], dim=0)

            # Extract user and item indices
            user_indices = batch_data["users"]
            item_indices = batch_data["pos_items"]

            user_indices = torch.IntTensor(user_indices).to("cuda:0")
            item_indices = torch.IntTensor(item_indices).to("cuda:0")

            # Compute user and item embeddings
            # user_embeddings = embeddings[user_indices]
            # item_embeddings = embeddings[item_indices]
            user_embeddings = self.user_embed(user_indices)
            item_embeddings = self.item_embed(item_indices)
            item_kg_embeddings = self.kg_entity_embed(item_indices)

            # Relation embeddings - assuming a single relation type for all interactions
            # Assuming index 0 is the interaction relation
            # relation_embedding = self.kg_relation_embed.weight[-1] # watched relation

            # Option 1: Dot Product Scoring Function (uncomment to use)
            # Compute scores for each user-item pair
            # batch_predictions = (user_embeddings + relation_embedding) @ item_embeddings.t()

            # Option 2: Euclidean Distance Scoring Function (uncomment to use)
            # Compute squared Euclidean distances for each user-item pair
            # batch_predictions = torch.sum((user_embeddings + relation_embedding - item_embeddings) ** 2, dim=1)

            # Option 3: Use the score used during training and with other libraries
            item_cf_embeddings = item_embeddings + item_kg_embeddings
            batch_predictions = torch.matmul(user_embeddings, item_cf_embeddings.t())
            return batch_predictions
        else:
            raise ValueError("Mode %s not supported" % mode)

    def _build_loss(self, u_e, pos_i_e, neg_i_e, h_e, r_e, pos_t_e, neg_t_e):
        kg_loss, kg_reg_loss = self._get_kg_loss(h_e, r_e, pos_t_e, neg_t_e)
        cf_loss, cf_reg_loss = self._get_cf_loss(u_e, pos_i_e, neg_i_e)

        base_loss = cf_loss
        kge_loss = kg_loss
        reg_loss = self.regs[0] * cf_reg_loss + self.regs[1] * kg_reg_loss
        total_loss = base_loss + kge_loss + reg_loss

        return total_loss, base_loss, kge_loss, reg_loss

    def _get_kg_loss(self, h_e, r_e, pos_t_e, neg_t_e):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = torch.sum((h_e + r_e - t_e) ** 2, dim=1, keepdim=True)
            return kg_score

        pos_kg_score = _get_kg_score(h_e, r_e, pos_t_e)
        neg_kg_score = _get_kg_score(h_e, r_e, neg_t_e)

        maxi = torch.log(torch.sigmoid(neg_kg_score - pos_kg_score))
        kg_loss = -torch.mean(maxi)
        kg_reg_loss = (
            torch.sum(h_e**2)
            + torch.sum(r_e**2)
            + torch.sum(pos_t_e**2)
            + torch.sum(neg_t_e**2)
        )

        return kg_loss, kg_reg_loss

    def _get_cf_loss(self, u_e, pos_i_e, neg_i_e):
        def _get_cf_score(u_e, i_e):
            cf_score = torch.sum(u_e * i_e, dim=1)
            return cf_score

        pos_cf_score = _get_cf_score(u_e, pos_i_e)
        neg_cf_score = _get_cf_score(u_e, neg_i_e)

        maxi = torch.log(1e-10 + torch.sigmoid(pos_cf_score - neg_cf_score))
        cf_loss = -torch.mean(maxi)
        cf_reg_loss = torch.sum(u_e**2) + torch.sum(pos_i_e**2) + torch.sum(neg_i_e**2)

        return cf_loss, cf_reg_loss
