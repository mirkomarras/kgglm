import argparse
import os

from kgglm.utils import get_weight_ckpt_dir, get_weight_dir

MODEL = 'CFKG'

def parse_args():
    parser = argparse.ArgumentParser(description=f"Run {MODEL}")
    parser.add_argument('--dataset', nargs='?', default='ml1m',
                        help='Choose a dataset from {ml1m, lfm1m}')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity / relation Embedding size.')
    parser.add_argument('--train_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=2,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')
    parser.add_argument('--print_every', type=int, default=1,
                        help='Iter interval of printing loss.')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='After how many epochs save ckpt')
    parser.add_argument('--Ks', nargs='?', default='[10]',
                        help='Calculate metric@K when evaluating.')

    args, unk = parser.parse_known_args()
    args.Ks = eval(args.Ks)
    args.model_type = MODEL
    args.weight_dir = get_weight_dir(MODEL, args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir(MODEL, args.dataset)
    log_dir = os.path.join('logs', args.dataset, args.model_type)
    args.log_dir = log_dir
    return args


