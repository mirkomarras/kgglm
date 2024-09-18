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
from helper.evaluation.eval_utils import save_topks_items_results
from helper.evaluation.utility_metrics import NDCG, PRECISION, RECALL
from helper.logging.log_helper import create_log_id, logging_config
from helper.models.model_utils import (EarlyStopping, compute_topks,
                                       load_model, logging_metrics, save_model)
from helper.models.traditional.BPRMF.bprmf import BPRMF
from helper.models.traditional.BPRMF.dataloader import DataLoaderBPRMF
from helper.models.traditional.BPRMF.parser import parse_bprmf_args
from helper.utils import SEED


def evaluate(model, dataloader, Ks, device):
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

    metrics_dict = {k: defaultdict(list) for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, is_train=False)  # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            topk_items_dict = compute_topks(batch_scores, train_user_dict, valid_user_dict, test_user_dict, batch_user_ids.cpu().numpy(),
                                          item_ids.cpu().numpy(), Ks)
            avg_metrics_dict = {k: evaluate_rec_quality(dataloader.data_name, topk_items_dict, test_user_dict, k)[1] for k in Ks}
            for k in Ks:
                for m in avg_metrics_dict[k].keys():
                    metrics_dict[k][m].append(avg_metrics_dict[k][m])
            pbar.update(1)

    for k in Ks:
        for m in metrics_dict[k].keys():
            metrics_dict[k][m] = np.array(metrics_dict[k][m]).mean()

    return topk_items_dict, metrics_dict


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
    data = DataLoaderBPRMF(args, logging)

    # Initialize model
    user_pre_embed = torch.tensor(data.user_pre_embed) if args.use_pretrain == 1 else None
    item_pre_embed = torch.tensor(data.item_pre_embed) if args.use_pretrain == 1 else None
    model = BPRMF(args, data.n_users, data.n_items, user_pre_embed, item_pre_embed).to(device)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)
    logging.info(model)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize early stopping
    early_stopper = EarlyStopping(args.stopping_steps, verbose=True)
    best_ndcg_value = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(1, args.n_epoch + 1):
        model.train()  # Set model to training mode
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        time1 = time()
        for iter in range(1, n_batch + 1):
            time2 = time()
            batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(data.train_user_dict,
                                                                                data.train_batch_size)
            batch_user, batch_pos_item, batch_neg_item = batch_user.to(device), batch_pos_item.to(
                device), batch_neg_item.to(device)

            optimizer.zero_grad()
            batch_loss = model(batch_user, batch_pos_item, batch_neg_item, is_train=True)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info(f'CF Training: Epoch {epoch:04d} Iter {iter:04d} / {n_batch:04d} | '
                             f'Time {time() - time2:.1f}s | Iter Loss {batch_loss.item():.4f} | '
                             f'Iter Mean Loss {total_loss / iter:.4f}')

        logging.info(f'CF Training: Epoch {epoch:04d} Total Iter {n_batch:04d} | Total Time {time() - time1:.1f}s | '
                     f'Iter Mean Loss {total_loss / n_batch:.4f}')

        # Evaluation and logging
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
    #Load model from best epoch
    model = load_model(model, os.path.join(args.weight_dir, f'{model.name}_epoch_{best_epoch}_e{args.embed_dim}_bs{args.train_batch_size}_lr{args.lr}.pth'))
    topk_items_dict, _ = evaluate(model, data, args.Ks, device)
    save_topks_items_results(args.dataset, model.name, topk_items_dict, k=args.Ks[0])


# fare un check
# def predict(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # load data
#     data = DataLoaderBPRMF(args, logging)

#     # load model
#     model = BPRMF(args, data.n_users, data.n_items)
#     model = load_model(model, args.pretrain_model_path)
#     model.to(device)

#     # predict
#     Ks = args.Ks
#     k_min = min(Ks)
#     k_max = max(Ks)

#     cf_scores, metrics_dict = evaluate(model, data, Ks, device)
#     np.save(args.save_dir + 'cf_scores.npy', cf_scores)
#     print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
#         metrics_dict[k_min][PRECISION], metrics_dict[k_max][PRECISION], metrics_dict[k_min][RECALL],
#         metrics_dict[k_max][RECALL], metrics_dict[k_min][NDCG], metrics_dict[k_max][NDCG]))


if __name__ == '__main__':
    args = parse_bprmf_args()
    train(args)
    # predict(args)
