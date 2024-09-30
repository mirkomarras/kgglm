import heapq
import logging
import os
import random
from collections import defaultdict
from time import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from helper.evaluation.eval_metrics import evaluate_rec_quality
from helper.evaluation.utility_metrics import NDCG
from helper.logging.log_helper import create_log_id, logging_config
from helper.models.knowledge_aware.CFKG.CFKG import CFKG
from helper.models.knowledge_aware.CFKG.dataloader import DataLoaderCFKG
from helper.models.knowledge_aware.CFKG.parser import parse_args
from helper.models.model_utils import (EarlyStopping, compute_topks,
                                       load_model, logging_metrics, save_model)
from helper.utils import SEED


def evaluate(model, dataloader, K, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    valid_user_dict = dataloader.valid_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    metrics_dict = {k: defaultdict(list) for k in K}
    relation_u2i_id = torch.LongTensor([dataloader.relation_u2i_id]).to(args.device)

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, relation_u2i_id, is_train=False)  # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            topk_items_dict = compute_topks(batch_scores, train_user_dict, valid_user_dict, test_user_dict, batch_user_ids.cpu().numpy(),item_ids.cpu().numpy(), K)
            avg_metrics_dict = {k: evaluate_rec_quality(dataloader.data_name, topk_items_dict, test_user_dict, k)[1] for k in K}
            for k in K:
                for m in avg_metrics_dict[k].keys():
                    metrics_dict[k][m].append(avg_metrics_dict[k][m])
            pbar.update(1)
    for k in K:
        for m in metrics_dict[k].keys():
            metrics_dict[k][m] = np.array(metrics_dict[k][m]).mean()

    return topk_items_dict, metrics_dict

def train_epoch(model, optimizer, data_generator, epoch, args):
    total_loss = 0.0
    n_batch = data_generator.n_kg_train // data_generator.train_batch_size + 1
    for iter in range(1, n_batch):
        batch_head, batch_relation, batch_pos_tail, batch_neg_tail = data_generator.generate_kg_batch(data_generator.train_kg_dict, data_generator.train_batch_size, data_generator.n_users_entities)
        batch_head = batch_head.to(args.device)
        batch_relation = batch_relation.to(args.device)
        batch_pos_tail = batch_pos_tail.to(args.device)
        batch_neg_tail = batch_neg_tail.to(args.device)
        batch_loss = model(batch_head, batch_relation, batch_pos_tail, batch_neg_tail, is_train=True)

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += batch_loss.item()

        if (iter % args.print_every) == 0:
            logging.info(f'KG Training: Epoch {epoch:04d} Iter {iter:04d} / {n_batch:04d} | '
                         f'Iter Loss {batch_loss.item():.4f} | '
                         f'Iter Mean Loss {total_loss / iter:.4f}')
    return total_loss


def print_training_info(epoch,training_time, epoch_loss):
    logging.info(f"Epoch {epoch} | Training Time {round(training_time,2)}s | Total Epoch Loss {epoch_loss}")


def ranklist_by_heapq(user_negatives, rating, K):
    item_score = {i: rating[i] for i in user_negatives}
    K_max_item_score = heapq.nlargest(K, item_score, key=item_score.get)
    return K_max_item_score


def train(args):
    # seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name=f'log{log_save_id}', no_console=False)
    logging.info(args)

    # GPU / CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device=device

    # load data
    dataset_obj = DataLoaderCFKG(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(dataset_obj.user_pre_embed)
        item_pre_embed = torch.tensor(dataset_obj.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = CFKG(args, dataset_obj.n_users, dataset_obj.n_entities, dataset_obj.n_relations, user_pre_embed, item_pre_embed)

    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    model.to(device)

    logging.info(model)
    best_ndcg_value = 0.0
    best_epoch=0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in tqdm(range(args.epoch)):
        #todo: modificare cfkg per scrivere model.train() (guarda calc_loss(...is_train=True))
        t1 = time()
        # Phase 1: CF training
        epoch_loss = train_epoch(model, optimizer, dataset_obj, epoch, args)

        print_training_info(epoch,time()-t1, epoch_loss)
        assert np.isnan(epoch_loss) == False

        # Phase 3: Test
        # Testing and performance logging

        _, metrics_dict = evaluate(model, dataset_obj, args.Ks, args.device) # evaluate at the end of each epoch.. to modify!
        logging_metrics(epoch, metrics_dict, args.Ks)
        is_best = metrics_dict[args.Ks[0]][NDCG] > best_ndcg_value
        best_ndcg_value = max(metrics_dict[args.Ks[0]][NDCG], best_ndcg_value)

        if is_best:
            save_model(model, args.weight_dir, args, epoch, best_epoch)
            best_epoch = epoch

        early_stopping(metrics_dict[args.Ks[0]][NDCG])

        if early_stopping.early_stop:
            logging.info('Early stopping triggered. Stopping training.')
            break

        # Optional: Save model and metrics at each epoch or at specific intervals
        if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
            torch.save(model.state_dict(), os.path.join(args.weight_dir_ckpt,f'{args.model_type}_epoch_{epoch}_bs{args.train_batch_size}_lr{args.lr}.pth'))
    # Final model save and cleanup
    torch.save(model.state_dict(), os.path.join(args.weight_dir,f'{args.model_type}_epoch_{epoch}_bs{args.train_batch_size}_lr{args.lr}.pth'))
    logging.info(f'Best evaluation results at epoch {early_stopping.best_epoch} with NDCG: {early_stopping.best_score:.4f}')


# def predict(args): 
#     # GPU / CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # load data
#     data = DataLoaderECFKG(args, logging)

#     # load model
#     model = ECFKG(args, data.n_users, data.n_entities, data.n_relations)
#     model = load_model(model, args.pretrain_model_path)
#     model.to(device)

#     # predict
#     Ks = eval(args.Ks)
#     k_min = min(Ks)
#     k_max = max(Ks)

#     cf_scores, metrics_dict = evaluate(model, data, Ks, device)
#     np.save(args.save_dir + 'cf_scores.npy', cf_scores)
#     print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
#         metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))



if __name__ == '__main__':
    args = parse_args()
    train(args)
    # predict(args)


