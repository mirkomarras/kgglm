import argparse
import os

from pathlm.evaluation.eval_utils import get_result_dir
from pathlm.utils import SEED, get_weight_ckpt_dir, get_weight_dir

MODEL = "bprmf"


def parse_bprmf_args():
    parser = argparse.ArgumentParser(description="Run BPRMF.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="amazon-book",
        help="Choose a dataset from {yelp2018, last-fm, amazon-book}",
    )
    parser.add_argument(
        "--data_dir", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument(
        "--use_pretrain",
        type=int,
        default=0,
        help="0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.",
    )
    parser.add_argument(
        "--pretrain_embedding_dir",
        nargs="?",
        default=f"weights/ml1m/{MODEL}/",
        help="Path of learned embeddings.",
    )
    parser.add_argument(
        "--pretrain_model_path",
        nargs="?",
        default=f"weights/ml1m/{MODEL}/{MODEL}.pth",
        help="Path of stored model.",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=64, help="User / item Embedding size."
    )
    parser.add_argument(
        "--l2loss_lambda",
        type=float,
        default=1e-5,
        help="Lambda when calculating CF l2 loss.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=1024, help="Train batch size."
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10000,
        help="Test batch size (the number of users to test every batch).",
    )

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--n_epoch", type=int, default=300, help="Number of epoch.")
    parser.add_argument(
        "--stopping_steps",
        type=int,
        default=3,
        help="Number of epoch for early stopping",
    )

    parser.add_argument(
        "--print_every", type=int, default=50, help="Iter interval of printing loss."
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=10,
        help="Epoch interval of evaluating CF.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=20, help="After how many epochs save ckpt"
    )
    parser.add_argument(
        "--Ks", nargs="?", default="[10]", help="Calculate metric@K when evaluating."
    )

    args = parser.parse_args()

    args.Ks = eval(args.Ks)
    log_dir = os.path.join("logs", args.dataset, MODEL)
    args.weight_dir = get_weight_dir("bprmf", args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir("bprmf", args.dataset)
    args.result_dir = get_result_dir("bprmf", args.dataset)
    args.log_dir = log_dir
    return args
