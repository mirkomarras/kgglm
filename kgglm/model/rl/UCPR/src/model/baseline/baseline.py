from collections import namedtuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from kgglm.data.dataset.datasets_utils import load_kg  # , load_dataset
from kgglm.data.knowledge_graph.kg_macros import SELF_LOOP, USER
from kgglm.model.rl.UCPR.src.model.lstm_base.backbone_lstm import KGState_LSTM
from kgglm.model.rl.UCPR.src.model.lstm_base.model_kg import KG_KGE  # , RW_KGE
from kgglm.model.rl.UCPR.src.model.lstm_base.model_kg_pre import (
    KG_KGE_pretrained,
)  # , RW_KGE_pretrained
from kgglm.model.rl.UCPR.utils import PADDING

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class ActorCritic(nn.Module):
    def __init__(
        self,
        args,
        user_triplet_set,
        rela_2_index,
        act_dim,
        gamma=0.99,
        hidden_sizes=[512, 256],
    ):
        super(ActorCritic, self).__init__()
        self.args = args
        self.act_dim = act_dim
        self.device = args.device
        self.sub_batch_size = args.sub_batch_size
        self.gamma = gamma
        self.p_hop = args.p_hop
        self.hidden_sizes = hidden_sizes
        self.n_memory = args.n_memory
        self.kg = load_kg(args.dataset)
        # self.gradient_plot_save = args.gradient_plot_save

        self.embed_size = args.embed_size

        self.user_triplet_set = user_triplet_set
        self.rela_2_index = rela_2_index

        if self.args.envir == "p1":
            # self._get_next_node_type_meta
            raise NotImplementedError()
            # if self.args.KGE_pretrained:
            #     self.kg_emb = RW_KGE_pretrained(args)
            # else:
            #     self.kg_emb = RW_KGE(args)

        elif self.args.envir == "p2":
            # self._get_next_node_type_graph
            if self.args.KGE_pretrained:
                self.kg_emb = KG_KGE_pretrained(args)
            else:
                self.kg_emb = KG_KGE(args)

            # dataset = load_dataset(args.dataset)

        self.bulid_model_rl()

    def _get_next_node_type(self, curr_node_type, relation, next_node_id):
        return self.kg(curr_node_type, next_node_id, relation)

    def bulid_model_rl(self):
        self.state_lstm = KGState_LSTM(self.args, history_len=1)

        self.transfor_state = nn.Linear(2 * self.embed_size, 2 * self.embed_size)
        self.state_tr_query = nn.Linear(self.embed_size * 3, self.embed_size)

        self.l1 = nn.Linear(4 * self.embed_size, 2 * self.embed_size)
        self.l2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.actor = nn.Linear(2 * self.embed_size, self.act_dim)
        self.critic = nn.Linear(2 * self.embed_size, 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs):
        state, _, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        state = state.squeeze(1)
        x = self.l1(state)
        actor_logits = self.actor(x)

        probs = actor_logits.masked_fill(~act_mask, value=torch.tensor(-1e10))
        act_probs = F.softmax(probs, dim=-1)

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_state, batch_next_action_emb, batch_act_mask, device):
        act_mask = torch.BoolTensor(batch_act_mask).to(
            device
        )  # Tensor of [bs, act_dim]
        probs, value = self(
            (batch_state, batch_next_action_emb, act_mask)
        )  # act_probs: [bs, act_dim], state_value: [bs, 1]

        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False

        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())

        return acts.cpu().numpy().tolist()

    def reset(self, uids=None):
        self.uids = [uid for uid in uids for _ in range(1)]

    def update_path_info(self, up_date_hop):
        new_uids = []

        for row in up_date_hop:
            new_uids.append(self.uids[row])

        self.uids = new_uids

    def generate_st_emb(self, batch_path, up_date_hop=None):
        if up_date_hop is not None:
            self.update_path_info(up_date_hop)
        state_output = torch.cat(
            [
                self._get_state_update(index, path).unsqueeze(0)
                for index, path in enumerate(batch_path)
            ],
            0,
        )
        return state_output

    def _get_state_update(self, index, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""

        user_embed = self.kg_emb.lookup_emb(
            USER, type_index=torch.LongTensor([path[0][-1]]).to(self.device)
        )[0].unsqueeze(0)
        if len(path) == 1:
            curr_node_embed = user_embed
            zero_embed = torch.zeros(self.embed_size).to(self.device).unsqueeze(0)

            st_emb = torch.cat([user_embed, user_embed, zero_embed, zero_embed], -1)

        else:
            older_relation, last_node_type, last_node_id = path[-2]
            last_relation, curr_node_type, curr_node_id = path[-1]
            curr_node_embed = self.kg_emb.lookup_emb(
                curr_node_type,
                type_index=torch.LongTensor([curr_node_id]).to(self.device),
            )[0].unsqueeze(0)
            last_node_embed = self.kg_emb.lookup_emb(
                last_node_type,
                type_index=torch.LongTensor([last_node_id]).to(self.device),
            )[0].unsqueeze(0)

            last_relation_embed = self.kg_emb.lookup_rela_emb(last_relation)[
                0
            ].unsqueeze(0)
            # st_emb = self.action_encoder(last_relation_embed, curr_node_embed)
            st_emb = torch.cat(
                [user_embed, curr_node_embed, last_node_embed, last_relation_embed], -1
            )

        return st_emb

    def action_encoder(self, relation_emb, entitiy_emb):
        action_embedding = torch.cat([relation_emb, entitiy_emb], -1)
        return action_embedding

    def generate_act_emb(self, batch_path, batch_curr_actions):
        return None

    def _get_actions(self, index, curr_path, curr_actions):
        last_relation, curr_node_type, curr_node_id = curr_path[-1]
        entities_embs = []
        relation_embs = []

        for action_set in curr_actions:
            if action_set[0] == SELF_LOOP:
                next_node_type = curr_node_type
            else:
                next_node_type = self._get_next_node_type(
                    curr_node_type, action_set[0], action_set[1]
                )
            enti_emb = self.kg_emb.lookup_emb(
                next_node_type,
                type_index=torch.LongTensor([action_set[1]]).to(self.device),
            )
            entities_embs.append(enti_emb)
            rela_emb = self.kg_emb.lookup_rela_emb(action_set[0])
            relation_embs.append(rela_emb)

        pad_emb = self.kg_emb.lookup_rela_emb(PADDING)
        for _ in range(self.act_dim - len(entities_embs)):
            entities_embs.append(pad_emb)
            relation_embs.append(pad_emb)

        enti_emb = torch.cat(entities_embs, 0)
        rela_emb = torch.cat(relation_embs, 0)

        next_action_state = torch.cat([enti_emb, rela_emb], -1)

        return next_action_state

    def update(self, optimizer, env_model, device, ent_weight, step):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += (
                self.gamma * batch_rewards[:, num_steps - i]
            )

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[
                i
            ]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()
