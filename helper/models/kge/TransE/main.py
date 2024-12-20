"""
TRAIN + EVALUATE
"""
import logging
import os
import random
from time import time

import numpy as np
import torch
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from tqdm.autonotebook import tqdm

from helper.data_mappers.mapper_kge import get_watched_relation_idx
from helper.datasets.datasets_utils import get_set
from helper.evaluation.eval_metrics import evaluate_rec_quality
from helper.logging.log_helper import create_log_id, logging_config
from helper.models.kge.TransE.parser_transe import parse_args
from helper.models.kge.TransE.transe import TransE
from helper.models.kge.utils import (build_kg_triplets, get_log_dir,
                                     get_set_lp, get_test_uids,
                                     get_users_positives,
                                     get_users_positives_lp, load_kg,
                                     metrics_lp, remap_topks2datasetid,load_kg_lp)
from helper.models.model_utils import EarlyStopping, logging_metrics
from helper.utils import SEED

method_name = "TransE"

"""Utils"""


def initialize_model(kg_train, b_size, emb_dim, weight_decay, margin, lr, use_cuda):
    """Define Model"""
    model = TransE(kg_train.n_ent, kg_train.n_rel, margin, emb_dim)
    """Define the torch optimizer to be used"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    """Define negative sampler"""
    sampler = BernoulliNegativeSampler(kg_train)
    """Define Dataloader"""
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=use_cuda)
    return model, optimizer, sampler, dataloader


def train_epoch(model, sampler, optimizer, dataloader, epoch, args):
    running_loss = 0.0
    model.train()
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r, n_neg=5)

        optimizer.zero_grad()

        """forward + backward + optimize"""
        pos, neg = model.forward(h, t, n_h, n_t, r)
        loss = model.loss(pos, neg, torch.ones_like(pos))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i % args.print_every) == 0:
            # per debugging metti logging.warning cosi appare in stdout
            logging.info(
                f'KG Training: Epoch {epoch:04d} Iter {i:04d} / {args.epoch:04d} | Iter Loss {running_loss:.4f}')

    return running_loss


def print_training_info(epoch, train_time, loss):
    logging.info(f"Epoch {epoch} [{train_time:.1f}s]: Loss: {loss:.5f}")

def batch_array(arr, batch_size):
    extra = (len(arr) % batch_size) != 0
    batched_array = [arr[i * batch_size:(i + 1) * args.batch_size] for i in range((len(arr) // batch_size) + extra)]
    return batched_array

def evaluate_model(model, args):
    if not args.lp:
        """Normalize parameters after each training"""
        model.normalize_parameters()
        """Get Watched Relation"""
        WATCHED = get_watched_relation_idx(args.dataset)
        """Load Pids identifiers"""
        if args.dataset in ['ml1m', 'lfm1m']:
            pids_identifiers = np.load(f"{args.preprocessed_torchkge}/pids_identifiers_new.npy")
        else:
            pids_identifiers = np.load(os.path.join('data', args.dataset, 'mappings', 'pids_identifiers_new.npy'))
        kg_train = load_kg(args.dataset)
        remapped_pids = [kg_train.ent2ix[pid] for pid in pids_identifiers]
        """Get kg test uids"""
        uids = get_test_uids(args.dataset)
        """Get users_positives, pids the user has already interacted with"""
        users_positives = get_users_positives(args.dataset, 'recommendation')
        """Load Embeddings"""
        entities_emb, relations_emb = model.get_embeddings()
        products_emb = entities_emb[remapped_pids]
        """Learning To Rank"""
        top_k_recommendations = {}
        batched_uids = batch_array(uids, args.batch_size)
        for b_uids in tqdm(batched_uids, desc="Extract top-k recommendations for each user", position=1, leave=False):
            remapped_uids = [kg_train.ent2ix[uid] for uid in b_uids]
            user_embs = entities_emb[remapped_uids]
            user_rel_embs = user_embs + relations_emb[kg_train.rel2ix[WATCHED]]
            dot_prod = torch.mm(user_rel_embs.cpu(), products_emb.T.cpu())
            users_positives_mask = []
            for uid in b_uids:
                users_positives_mask.append([kg_train.ent2ix[pid] for pid in users_positives[uid]])
            history_u = torch.cat([torch.full((len(hist_iid),), i, dtype=int) for i, hist_iid in enumerate(users_positives_mask)])
            history_i = torch.LongTensor([x for p_mask in users_positives_mask for x in p_mask])
            dot_prod[(history_u, history_i)] = -torch.inf
            _, indexes = torch.topk(dot_prod, k=args.K, dim=1)
            for i, uid in enumerate(b_uids):
                top_k_recommendations[uid] = indexes[i].numpy()
        """Remap uid of top_k_recommendations to dataset id"""
        if args.dataset in ['ml1m','lfm1m']:
            top_k_recommendations = remap_topks2datasetid(args, top_k_recommendations)
        test_labels = get_set(args.dataset, 'test')
        _, avg_rec_quality_metrics = evaluate_rec_quality(args.dataset, top_k_recommendations, test_labels, args.K, method_name=method_name)
        return avg_rec_quality_metrics, top_k_recommendations
    else:
        kg_train = load_kg_lp(args.dataset, 'train')
        users_positives = get_users_positives_lp(args.dataset)
        test_labels = get_set_lp(args.dataset, 'test')
        """Generating Top k"""
        top_k = {}
        e1_values = [key[0] for key in test_labels.keys()]
        r_values = [key[1] for key in test_labels.keys()]
        e2_tensor = torch.IntTensor(list(kg_train.ent2ix.keys())).to(args.device)
        emb2entity = {value: key for key, value in kg_train.ent2ix.items()}
        for e1, r in zip(e1_values, r_values):
            e1_r = (int(e1), int(r))
            e1 = torch.IntTensor([e1]).to(args.device)
            r = torch.IntTensor([kg_train.rel2ix[r]]).to(args.device)
            scores = model.scoring_function(e1, e2_tensor, r)
            users_positives_mask = [int(kg_train.ent2ix[e2]) for e2 in users_positives[e1_r]]
            scores[users_positives_mask] = float('-inf')
            indexes = np.argsort(scores.cpu().detach().numpy())[::-1]
            top_k[e1_r] = indexes[:args.K]
            top_k[e1_r] = np.array([emb2entity[index] for index in top_k[e1_r]])
        metrics = metrics_lp(test_labels, top_k)
        return metrics, top_k


def train(args):
    """Set random seeds for reproducibility"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    """Setup logging"""
    log_dir = get_log_dir(method_name)
    log_save_id = create_log_id(log_dir)
    logging_config(folder=log_dir, name=f'log{log_save_id}', no_console=True)
    logging.info(args)

    """Setup device (GPU/CPU)"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.device = device

    """Load kg_train and initialize model"""
    if not args.lp:
        kg_train = load_kg(args.dataset)
    else:
        build_kg_triplets(args.dataset)
        kg_train = load_kg_lp(args.dataset, 'train')
    model, optimizer, sampler, dataloader = initialize_model(kg_train, args.batch_size, args.embed_size, args.weight_decay, args.margin, args.lr, args.use_cuda)

    """Move everything to MPS or CUDA or CPU if available"""
    model.to(args.device)

    """Training loop"""
    logging.info(model)
    early_stopping = EarlyStopping(patience=15, verbose=True)

    iterator = tqdm(range(args.epoch), unit='epoch')
    for epoch in iterator:
        t1 = time()
        """Phase 1: CF training"""
        running_loss = train_epoch(model, sampler, optimizer, dataloader, epoch, args)
        print_training_info(epoch, time() - t1, running_loss)
        assert np.isnan(running_loss) == False

        """
        Phase 2: Test
        Testing and performance logging
        """
        model.eval()
        test_metrics, topks = evaluate_model(model, args)
        logging_metrics(epoch, test_metrics, [str(args.K)])

        ndcg_value = test_metrics['ndcg']
        early_stopping(ndcg_value)

        if early_stopping.early_stop:
            logging.info('Early stopping triggered. Stopping training.')
            break

        """Optional: Save model and metrics at each epoch or at specific intervals"""
        if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
            if args.lp:
                torch.save(model.state_dict(), os.path.join(args.weight_dir_ckpt,
                                                            f'{method_name}_LinkPrediction_dataset_{args.dataset}_ndcg_{round(test_metrics["ndcg"], 2)}_mr_{round(test_metrics["mr"], 2)}_mrr_{round(test_metrics["mrr"], 2)}_hits@1_{round(test_metrics["hits@1"], 2)}_hits@3_{round(test_metrics["hits@3"], 2)}_hits@10_{round(test_metrics["hits@10"], 2)}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.weight_dir_ckpt,
                                                            f'{method_name}_dataset_{args.dataset}_ndcg_{round(test_metrics["ndcg"], 2)}_mrr_{round(test_metrics["mrr"], 2)}_prec_{round(test_metrics["precision"], 2)}_rec_{round(test_metrics["recall"], 2)}_ser_{round(test_metrics["serendipity"], 2)}_div_{round(test_metrics["diversity"], 2)}_nov_{round(test_metrics["novelty"], 2)}_cov_{round(test_metrics["coverage"], 2)}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))

        iterator.set_description('Epoch {} | mean loss: {:.5f}'.format(
            epoch + 1, running_loss / len(dataloader)))

    """Final model save and cleanup"""
    if args.lp:
        torch.save(model.state_dict(), os.path.join(args.weight_dir,
                                                    f'{method_name}_LinkPrediction_dataset_{args.dataset}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(args.weight_dir,
                                                    f'{method_name}_dataset_{args.dataset}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
    logging.info(
        f'Best evaluation results at epoch {early_stopping.best_epoch} with NDCG: {early_stopping.best_score:.4f}')


if __name__ == '__main__':
    args = parse_args()

    if args.task == 'train':
        train(args)
    else:  # args.task == 'evaluate'
        if not args.lp:
            kg_train = load_kg(args.dataset)
        else:
            # lp datasets should have: 'entities.dict', 'relations.dict', 'train.txt', 'valid.txt', 'test.txt'
            build_kg_triplets(args.dataset)
            kg_train = get_set_lp(args.dataset, 'train')
        model = TransE(kg_train.n_ent, kg_train.n_rel,
                       args.margin, args.embed_size)

        """Setup device (GPU/CPU)"""
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        args.device = device

        model.to(args.device)
        model.load_state_dict(torch.load(
            f"{args.weight_dir_ckpt}/{args.model_checkpoint}"))
        model.eval()
        evaluate_model(model, args)  # print metrics
