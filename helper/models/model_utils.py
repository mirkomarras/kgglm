import logging
import torch
import os
import numpy as np


class EarlyStopping:
    def __init__(self, patience, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
                            The training will stop if the metric does not improve for this number of epochs.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, ndcg_value):
        score = ndcg_value

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 1
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch += self.counter + 1
            self.counter = 0


def logging_metrics(epoch, metrics_dict, Ks):
    for k in Ks:
        if k not in metrics_dict:
            metrics_str = ', '.join([f'{key}: {metrics_dict[key]:.4f}' for key in metrics_dict.keys()])
            logging.info(f'Epoch {epoch} | K: {k} | {metrics_str}')
        else:
            #Log metric key metric value using join on dict
            metrics_str = ', '.join([f'{key}: {metrics_dict[k][key]:.4f}' for key in metrics_dict[k].keys()])
            logging.info(f'Epoch {epoch} | K: {k} | {metrics_str}')


def load_model(model, model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def save_model(model, model_dir, args, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = f'{model.name}_epoch_{current_epoch}_e{args.embed_dim}_bs{args.train_batch_size}_lr{args.lr}.pth'
    model_state_file = os.path.join(model_dir, filename)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_filename = f'{model.name}_epoch_{last_best_epoch}_e{args.embed_dim}_bs{args.train_batch_size}_lr{args.lr}.pth'
        old_model_state_file = os.path.join(model_dir, old_filename)
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))

def compute_topks(cf_scores, train_user_dict, valid_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for u in user_ids:
        train_pos_item_list = train_user_dict[u]
        valid_pos_item_list = valid_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[u][train_pos_item_list] = -np.inf
        cf_scores[u][valid_pos_item_list] = -np.inf
        test_pos_item_binary[u][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    topk_items_dict = {}  # Dictionary to store top-k items for each user
    maxK = max(Ks)
    for u in user_ids:
        topk_items = [item_ids[i] for i in rank_indices[u]][:maxK]  # Convert indices to real item IDs
        topk_items_dict[u] = topk_items
    return topk_items_dict




