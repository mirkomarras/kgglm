import logging
import os

import torch

from helper.evaluation.utility_metrics import NDCG, PRECISION, RECALL, MRR


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
