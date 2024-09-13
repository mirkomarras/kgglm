import torch
from torch import nn
from torch import cuda
import numpy as np
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

class TuckER(nn.Module):
    def __init__(self,emb_dim, num_entities, num_relations,args):
        super().__init__()
        # we suppose that the embedding dim is the same for entities and relations. This approach has been used for some experiments on the official paper (table 5)
        self.E = torch.nn.Embedding(num_entities, emb_dim)
        self.R = torch.nn.Embedding(num_relations, emb_dim)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (emb_dim, emb_dim, emb_dim)), dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(args.input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(args.hidden_dropout1)
        self.hidden_dropout2 = torch.nn.Dropout(args.hidden_dropout2)

        self.bn0 = torch.nn.BatchNorm1d(emb_dim)
        self.bn1 = torch.nn.BatchNorm1d(emb_dim)

        self.loss=torch.nn.BCELoss()

        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    
    def forward(self,e1_idx,_,r_idx):
        e1 = self.E(e1_idx)#128,64
        x = self.bn0(e1)#128,64
        x = self.input_dropout(x) #128,64
        x = x.view(-1, 1, e1.size(1)) # 128,1, 64

        r = self.R(r_idx)#128,64
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))# 128, 4096
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1)) #128,64,64
        W_mat = self.hidden_dropout1(W_mat) # 128, 64, 64

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1)) # 128, 64
        x = self.bn1(x)# 128, 64
        x = self.hidden_dropout2(x)# 128, 64
        x = torch.mm(x, self.E.weight.transpose(1,0))# 128, 19844 -> (128,64)*(64,19844)
        pred = torch.sigmoid(x) # 128, 19844
        return pred
