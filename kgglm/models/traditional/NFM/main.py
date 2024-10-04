import logging
import random
import sys
from time import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from tqdm import tqdm

from kgglm.evaluation.eval_metrics import evaluate_rec_quality
from kgglm.evaluation.utility_metrics import NDCG
from kgglm.logging.log_helper import create_log_id, logging_config,logging
from kgglm.models.model_utils import (EarlyStopping, compute_topks,
                                       load_model, logging_metrics, save_model)
from kgglm.models.traditional.NFM.dataloader import DataLoaderNFM
from kgglm.models.traditional.NFM.nfm import NFM
from kgglm.models.traditional.NFM.parser import parse_nfm_args
from kgglm.utils import SEED


def evaluate_batch(model, dataloader, user_ids, Ks):
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    feature_values = dataloader.generate_test_batch(user_ids)
    with torch.no_grad():
        scores = model(feature_values, is_train=False)              # (batch_size)

    rows = [user_idx_map[u] for u in np.repeat(user_ids, n_items).tolist()]
    cols = item_ids * n_users
    score_matrix = torch.Tensor(sp.coo_matrix((scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    metrics_dict = compute_topks(score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, Ks)

    score_matrix = score_matrix.numpy()
    return score_matrix, metrics_dict


def evaluate(model, dataloader, Ks, device, num_processes=4):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    valid_user_dict = dataloader.valid_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    cf_users = []
    cf_items = []
    cf_scores = []

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user in user_ids_batches:
            feature_values = dataloader.generate_test_batch(batch_user)
            feature_values = feature_values.to(device)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)            # (batch_size)

            cf_users.extend(np.repeat(batch_user, n_items).tolist())
            cf_items.extend(item_ids * len(batch_user))
            cf_scores.append(batch_scores.cpu())
            pbar.update(1)

    rows = [user_idx_map[u] for u in cf_users]
    cols = cf_items
    cf_scores = torch.cat(cf_scores)
    cf_score_matrix = torch.Tensor(sp.coo_matrix((cf_scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)

    # changed
    topk_items_dict = compute_topks(cf_score_matrix, train_user_dict, valid_user_dict, test_user_dict, user_ids, item_ids, Ks)
    avg_metrics_dict = {k: evaluate_rec_quality(dataloader.dataset_name, topk_items_dict, test_user_dict, k)[1] for k in Ks}
    cf_score_matrix = cf_score_matrix.numpy()
    return cf_score_matrix, avg_metrics_dict


def train(args):
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Setup logging
    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name=f'log{log_save_id}', no_console=False)
    logging.info(args)

    # Setup device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = DataLoaderNFM(args,logging)

    # Initialize model
    user_pre_embed = torch.tensor(data.user_pre_embed) if args.use_pretrain == 1 else None
    item_pre_embed = torch.tensor(data.item_pre_embed) if args.use_pretrain == 1 else None

    model = NFM(args, data.n_users, data.n_items, data.n_entities, user_pre_embed, item_pre_embed).to(device)

    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path, device)
    logging.info(model)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize early stopping
    early_stopper = EarlyStopping(args.stopping_steps, verbose=True)
    best_ndcg_value = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()
            pos_feature_values, neg_feature_values = data.generate_train_batch(data.train_user_dict)
            pos_feature_values = pos_feature_values.to(device)
            neg_feature_values = neg_feature_values.to(device)
            batch_loss = model.forward(pos_feature_values, neg_feature_values, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                #logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit(-1)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info(f'CF Training: Epoch {epoch:04d} Iter {iter:04d} / {n_batch:04d} | '
                             f'Time {time() - time2:.1f}s | Iter Loss {batch_loss.item():.4f} | '
                             f'Iter Mean Loss {total_loss / iter:.4f}')

        logging.info(f'CF Training: Epoch {epoch:04d} Total Iter {n_batch:04d} | Total Time {time() - time1:.1f}s | '
                     f'Iter Mean Loss {total_loss / n_batch:.4f}')

        if epoch % args.evaluate_every == 0 or epoch == args.n_epoch:
            _, metrics_dict = evaluate(model, data, args.Ks, device)
            is_best = metrics_dict[args.Ks[0]][NDCG] > best_ndcg_value
            best_ndcg_value = max(metrics_dict[args.Ks[0]][NDCG], best_ndcg_value)

            if is_best:
                save_model(model, args.weight_dir, args, epoch, best_epoch)
                best_epoch = epoch

            early_stopper(metrics_dict[args.Ks[0]][NDCG])
            if early_stopper.early_stop:
                logging.info('Early stopping triggered. Stopping training.')
                break

            logging_metrics(epoch, metrics_dict, args.Ks)

        if epoch % args.save_interval == 0:
            save_model(model, args.weight_dir_ckpt, args, epoch)

    # Final log for best metrics
    logging.info(f'Best evaluation results at epoch {best_epoch} with NDCG: {best_ndcg_value:.4f}')


# da fixare
# def predict(args):
#     # GPU / CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # load data
#     data = DataLoaderNFM(args)
#
#     # load model
#     model = NFM(args, data.n_users, data.n_items, data.n_entities)
#     model = load_model(model, args.pretrain_model_path)
#     model.to(device)
#
#     # predict
#     num_processes = args.test_cores
#     #if num_processes and num_processes > 1:
#     #    evaluate_func = evaluate_mp
#     #else:
#     evaluate_func = evaluate
#
#     Ks = eval(args.Ks)
#     k_min = min(Ks)
#     k_max = max(Ks)
#
#     cf_scores, metrics_dict = evaluate_func(model, data, Ks, num_processes, device)
#     np.save(args.save_dir + 'cf_scores.npy', cf_scores)
#     print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
#         metrics_dict[k_min][PRECISION], metrics_dict[k_max][PRECISION], metrics_dict[k_min][RECALL],
#         metrics_dict[k_max][RECALL], metrics_dict[k_min][NDCG], metrics_dict[k_max][NDCG]))



if __name__ == '__main__':
    args = parse_nfm_args()
    train(args)
    # predict # da fixare
