import heapq
import logging
import os
import random
from time import time

import numpy as np
import torch
from tqdm import tqdm

from kgglm.datasets.datasets_utils import get_user_negatives
from kgglm.evaluation.eval_metrics import evaluate_rec_quality
from kgglm.logging.log_helper import create_log_id, logging_config
from kgglm.models.knowledge_aware.CKE.CKE import CKE
from kgglm.models.knowledge_aware.CKE.dataloader import CKELoader
from kgglm.models.knowledge_aware.CKE.parser import parse_args
from kgglm.models.model_utils import EarlyStopping, logging_metrics
from kgglm.utils import SEED, get_data_dir


def initialize_model(args, dataset_obj):
    config = {
        'n_users': dataset_obj.n_users,
        'n_items': dataset_obj.n_items,
        'n_relations': dataset_obj.n_relations,
        'n_entities': dataset_obj.n_entities,
    }
    model = CKE(config, pretrain_data=None, args=args).to(args.device)
    return model



def train_epoch(model, data_generator, epoch, args):
    total_loss, total_base_loss, total_kge_loss, total_reg_loss = 0.0, 0.0, 0.0, 0.0
    n_batch = data_generator.n_train // args.batch_size + 1
    for iter in range(1, n_batch):
        batch_data = data_generator.generate_train_batch()
        batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train_step(batch_data)
        total_loss += batch_loss.item()
        total_base_loss += batch_base_loss.item()
        total_kge_loss += batch_kge_loss.item()
        total_reg_loss += batch_reg_loss.item()
        if (iter % args.print_every) == 0:
            logging.info(f'KG Training: Epoch {epoch:04d} Iter {iter:04d} / {n_batch:04d} | '
                         f'Iter Loss {batch_loss.item():.4f} | '
                         f'Iter Mean Loss {total_loss / iter:.4f}')
    return total_loss / n_batch, total_base_loss / n_batch, total_kge_loss / n_batch, total_reg_loss / n_batch

def print_training_info(epoch, train_time, avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss):
    logging.info(f"Epoch {epoch} [{train_time:.1f}s]: Average loss: {avg_loss:.5f} = "
          f"Base loss: {avg_base_loss:.5f} + KGE loss: {avg_kge_loss:.5f} + Reg loss: {avg_reg_loss:.5f}")


def ranklist_by_heapq(user_negatives, rating, K):
    item_score = {i: rating[i] for i in user_negatives}
    K_max_item_score = heapq.nlargest(K, item_score, key=item_score.get)
    return K_max_item_score

def evaluate_model(model, users_to_test, kgat_dataset, args):
    K = args.K
    u_batch_size = args.test_batch_size * 2
    n_test_users = len(users_to_test)
    n_user_batches = n_test_users // u_batch_size + 1
    topks = {}
    user_negatives = get_user_negatives(args.dataset)
    model.eval()  # Set the model to evaluation mode

    for u_batch_id in range(n_user_batches):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = users_to_test[start:end]
        item_batch = list(range(0, model.n_items))
        feed_dict = kgat_dataset.prepare_test_data(user_batch, item_batch)
        # Forward pass through the model
        rate_batch = model(feed_dict, 'eval')  # Assuming model's forward pass returns the required ratings.
        rate_batch = rate_batch.detach().cpu().numpy()  # Convert to numpy for subsequent operations

        for i, user in enumerate(user_batch):
            user_ratings = rate_batch[i]
            candidate_items = user_negatives[user]
            pids = ranklist_by_heapq(candidate_items, user_ratings, K)
            topks[user] = pids

    avg_metrics_dict = evaluate_rec_quality(args.dataset, topks, kgat_dataset.test_user_dict, K)[1]

    return avg_metrics_dict, topks

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    #train_cores = multiprocessing.cpu_count()

    dataset_obj = CKELoader(args=args, path=os.path.join(get_data_dir(args.dataset),'kgat'))
    model = initialize_model(args, dataset_obj)
    logging.info(model)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in tqdm(range(args.epoch)):
        t1 = time()
        # Phase 1: CF training
        avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss = train_epoch(model, dataset_obj, epoch, args)
        print_training_info(epoch, time() - t1, avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss)
        assert np.isnan(avg_loss) == False

        # Phase 3: Test
        # Testing and performance logging
        t2 = time()
        users_to_test = list(dataset_obj.test_user_dict.keys())
        test_metrics, topks = evaluate_model(model, users_to_test, dataset_obj, args)
        logging_metrics(epoch, test_metrics, [str(args.K)])

        ndcg_value = test_metrics['ndcg']
        early_stopping(ndcg_value)

        if early_stopping.early_stop:
            logging.info('Early stopping triggered. Stopping training.')
            break

        # Optional: Save model and metrics at each epoch or at specific intervals
        if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
            torch.save(model.state_dict(), os.path.join(args.weight_dir_ckpt, f'{args.model_type}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
    # Final model save and cleanup
    torch.save(model.state_dict(), os.path.join(args.weight_dir, f'{args.model_type}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
    logging.info(f'Best evaluation results at epoch {early_stopping.best_epoch} with NDCG: {early_stopping.best_score:.4f}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
