import argparse
import os
import sys

from helper.utils import get_weight_ckpt_dir, get_weight_dir

MODEL = 'cke'


def parse_args():
    parser = argparse.ArgumentParser(description=F"Run {MODEL}.")
    parser.add_argument('--dataset', nargs='?', default='ml1m',
                        help='Choose a dataset from {ml1m, lfm1m}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to the model weights")
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=120,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64,
                        help='KG Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='CF batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='test batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--K', type=int, default=10,
                        help='Topk size')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='After how many epochs save ckpt')
    parser.add_argument('--print_every', type=int, default=20,
                        help='Iter interval of printing loss.')
    parser.add_argument('--l1_flag', type=bool, default=True,
                        help='Flase: using the L2 norm, True: using the L1 norm.')

    args,unk = parser.parse_known_args()
    args.model_type = MODEL
    args.weight_dir = get_weight_dir(MODEL, args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir(MODEL, args.dataset)
    log_dir = os.path.join('logs', args.dataset, args.model_type)
    args.log_dir = log_dir
    return args
