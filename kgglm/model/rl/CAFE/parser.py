import argparse
import os

import torch

from kgglm.model.rl.CAFE.cafe_utils import CAFE, TMP_DIR
from kgglm.utils import get_weight_ckpt_dir, get_weight_dir


def parse_args():
    def boolean(x):
        return str(x).lower() == "true"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="lfm1m",
        help="dataset name. One of {ml1m, lfm1m}",
    )
    parser.add_argument(
        "--name", type=str, default="neural_symbolic_model", help="model name."
    )
    parser.add_argument("--seed", type=int, default=123, help="random seed.")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device.")
    # Hyperparameters for training neural-symbolic model.
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate.")
    parser.add_argument(
        "--steps_per_checkpoint",
        type=int,
        default=100,
        help="Number of steps for checkpoint.",
    )
    parser.add_argument(
        "--embed_size", type=int, default=100, help="KG embedding size."
    )
    parser.add_argument(
        "--deep_module", type=boolean, default=True, help="Use deep module or not"
    )
    parser.add_argument(
        "--use_dropout", type=boolean, default=True, help="use dropout or not."
    )
    parser.add_argument(
        "--rank_weight",
        type=float,
        default=10,
        help="weighting factor for ranking loss.",
    )
    parser.add_argument(
        "--topk_candidates",
        type=int,
        default=10,
        help="weighting factor for ranking loss.",
    )

    # Hyperparameters for execute neural programs (inference).
    parser.add_argument(
        "--k", type=int, default=10, help="size of recommendation list."
    )
    parser.add_argument(
        "--sample_size", type=int, default=50, help="sample size for model."
    )
    parser.add_argument(
        "--do_infer",
        type=boolean,
        default=True,
        help="whether to infer paths after training.",
    )
    parser.add_argument(
        "--do_execute",
        type=boolean,
        default=True,
        help="whether to execute neural programs.",
    )
    parser.add_argument(
        "--do_validation", type=bool, default=True, help="Whether to perform validation"
    )
    parser.add_argument(
        "--save_interval", default=10, type=int, help="Interval to save model weights."
    )
    args = parser.parse_args()

    # This is model directory.
    args.log_dir = f"{TMP_DIR[args.dataset]}/{args.name}"
    args.weight_dir = get_weight_dir(CAFE, args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir(CAFE, args.dataset)
    # This is the checkpoint name of the trained neural-symbolic model.
    args.symbolic_model = (
        f"{args.weight_dir_ckpt}/symbolic_model_epoch{args.epochs}.ckpt"
    )

    # This is the filename of the paths inferred by the trained neural-symbolic model.
    args.infer_path_data = f"{args.weight_dir}/infer_path_data.pkl"

    # Set GPU device.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False

    return args
