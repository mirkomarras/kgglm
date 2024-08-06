import argparse
import os
import sys

from pathlm.models.kge_rec.utils import get_weight_dir, get_weight_ckpt_dir,get_preprocessed_torchkge_path

MODEL = 'TuckER'
def parse_args():
    parser = argparse.ArgumentParser(description=F"Run {MODEL}.")
    parser.add_argument('--dataset', nargs='?', default='ml1m',
                        help='Choose a dataset from {ml1m, lfm1m}')
    parser.add_argument('--epoch', type=int, default=120,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--weight_decay', type=float, default=1.0,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--K', type=int, default=10,
                        help='Topk size')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='After how many epochs save ckpt')
    parser.add_argument('--use_cuda', type=str, default='None',
                        help='use cuda for dataloader transE, all or none')
    parser.add_argument('--print_every', type=int, default=1, 
                        help='Iter interval of printing loss.')
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    
    parser.add_argument('--task', type=str, default='train', 
                        help='Train or evaluate on previous weights? specify train or evaluate')
    parser.add_argument('--model_checkpoint', type=str, default='', 
                        help='specify the name of the checkpoint .pth')
    """Link Prediction Task"""
    parser.add_argument('--lp', type=bool, default=False,
                        help='Train and evaluate on lp dataset')


    args = parser.parse_args()

    args.preprocessed_torchkge=get_preprocessed_torchkge_path(args.dataset)
    args.model_type = MODEL
    args.weight_dir = get_weight_dir(MODEL, args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir(MODEL, args.dataset)
    log_dir = os.path.join('logs', args.dataset, args.model_type)
    args.log_dir = log_dir
    return args