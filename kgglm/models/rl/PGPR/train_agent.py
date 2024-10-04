import copy
import json
import warnings
import random
import torch
from collections import defaultdict
import numpy as np
import logging
import os
from os import makedirs
import torch.optim as optim
from kgglm.utils import SEED
from kgglm.knowledge_graphs.kg_macros import USER
from kgglm.models.rl.PGPR.actor_critic import ACDataLoader, ActorCritic
from kgglm.models.rl.PGPR.kg_env import BatchKGEnvironment
from kgglm.models.rl.PGPR.parser import parser_pgpr_train
from kgglm.models.rl.PGPR.pgpr_utils import TMP_DIR,HPARAMS_FILE
from kgglm.utils import get_weight_ckpt_dir, get_weight_dir
from kgglm.logging.log_helper import get_logger

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = None

def train(args):
    # check how datasets are loaded by BatchKGEnvironment
    train_env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,state_history=args.state_history)
    valid_env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,state_history=args.state_history)
    
    train_uids = list(train_env.kg(USER).keys())
    valid_uids = list(valid_env.kg(USER).keys())

    train_dataloader = ACDataLoader(train_uids, args.batch_size)
    valid_dataloader = ACDataLoader(valid_uids, args.batch_size)

    model = ActorCritic(train_env.state_dim, train_env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model.load_state_dict(model_sd)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    metrics=defaultdict(list)
    loaders = {'train': train_dataloader,'valid': valid_dataloader}
    envs = {'train': train_env,'valid':valid_env}
    step_counter = {'train': 0,'valid':0}
    uids_split = {'train' :train_uids,'valid':valid_uids}

    first_iterate = True
    model.train()
    start = 0
    for epoch in range(1, args.epochs + 1):
        splits_to_compute = list(loaders.items())
        if first_iterate:
            first_iterate = False
            splits_to_compute.insert(0, ('valid', valid_dataloader))        
        for split_name, dataloader in splits_to_compute:
            if split_name == 'valid' and epoch%10 != 0:
                continue            
            if split_name == 'valid':
                model.eval()
            else:
                model.train()
            dataloader.reset()
            env = envs[split_name]
            uids = uids_split[split_name]
            iter_counter = 0
            ### Start epoch ###
            dataloader.reset()
            while dataloader.has_next():
                batch_uids = dataloader.get_batch()
                ### Start batch episodes ###
                batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
                done = False
                while not done:
                    batch_act_mask = env.batch_action_mask(dropout=args.act_dropout) # numpy array of size [bs, act_dim]
                    batch_act_idx = model.select_action(batch_state, batch_act_mask, args.device)  # int
                    batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                    model.rewards.append(batch_reward)
                ### End of episodes ###
                if split_name == 'train':
                    lr = args.lr * max(1e-4, 1.0 - float(step_counter[split_name]) / (args.epochs * len(uids) / args.batch_size))
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                # Update policy
                total_reward = np.sum(model.rewards)
                loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight)
                cur_metrics = { f'{split_name}_loss':loss,
                                f'{split_name}_ploss':ploss,
                                f'{split_name}_vloss':vloss,
                                f'{split_name}_entropy':eloss,
                                f'{split_name}_reward':total_reward,
                                f'{split_name}_iter': step_counter[split_name]}
                
                for k,v in cur_metrics.items():
                    metrics[k].append(v)
                
                step_counter[split_name] += 1
                iter_counter += 1

            cur_metrics = [f'{split_name}_epoch']
            cur_metrics.extend([
                f'{split_name}_loss',
                f'{split_name}_ploss',
                f'{split_name}_vloss',
                f'{split_name}_entropy',
                f'{split_name}_reward',
                ])

            for k in cur_metrics[1:]:
                logging.info(f'avg_{k}', sum(metrics[k][-iter_counter:])/max(iter_counter, 1))
            #metrics[f'avg_{split_name}_reward'][-1] /= args.batch_size 

            logging.info(f'{split_name}_epoch {epoch}')
            cur_metrics.append(f'std_{split_name}_reward')
            logging.info(f'std_{split_name}_reward ', np.std(metrics[f'{split_name}_reward'][-iter_counter:]))

            info = ""
            for k in cur_metrics:
                if len(metrics[k])!=0:
                    if isinstance(metrics[k],float):
                        x = f'{round(metrics[k][-1],5)}'
                    else:
                        x = metrics[k][-1]
                    info = info + f'| {k}={x}'
            logger.info(info)

        ### END of epoch ###
        if epoch % 1 == 0:
            policy_file = f'{args.weight_dir_ckpt}/policy_model_epoch_{epoch}.ckpt'
            logger.info(f"Save models to {policy_file}")
            torch.save(model.state_dict(), policy_file)
    makedirs(args.dataset)

def main():
    args=parser_pgpr_train()
    os.makedirs(TMP_DIR[args.dataset], exist_ok=True)
    with open(os.path.join(TMP_DIR[args.dataset],HPARAMS_FILE), 'w') as f:
        args_dict = dict()
        for x,y in copy.deepcopy(args._get_kwargs()):
            args_dict[x] = y
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict,f)

    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    args.weight_dir = get_weight_dir("pgpr", args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir("pgpr", args.dataset)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train(args)

if __name__ == '__main__':
    main()
