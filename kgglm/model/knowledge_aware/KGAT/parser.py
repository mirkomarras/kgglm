import argparse
import os
import sys

from pathlm.utils import get_weight_ckpt_dir, get_weight_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="ml1m",
        help="Choose a dataset from {ml1m, lfm1m}",
    )
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Path to the model weights",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Interval of evaluation."
    )
    parser.add_argument("--epoch", type=int, default=120, help="Number of epoch.")
    parser.add_argument("--embed_size", type=int, default=64, help="CF Embedding size.")
    parser.add_argument("--kge_size", type=int, default=64, help="KG Embedding size.")
    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[64,32,16]",
        help="Output sizes of every layer",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="CF batch size.")
    parser.add_argument(
        "--test_batch_size", type=int, default=1024, help="test batch size."
    )
    parser.add_argument(
        "--batch_size_kg", type=int, default=2048, help="KG batch size."
    )
    parser.add_argument(
        "--regs",
        nargs="?",
        default="[1e-5,1e-5]",
        help="Regularization for user and item embeddings.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--with_replacement",
        type=bool,
        default=False,
        help="whether to use replacement for sampling",
    )
    parser.add_argument(
        "--model_type",
        nargs="?",
        default="kgat",
        help="Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.",
    )
    parser.add_argument(
        "--adj_type",
        nargs="?",
        default="si",
        help="Specify the type of the adjacency (laplacian) matrix from {bi, si}.",
    )
    parser.add_argument(
        "--alg_type",
        nargs="?",
        default="bi",
        help="Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.",
    )
    parser.add_argument(
        "--adj_uni_type",
        nargs="?",
        default="sum",
        help="Specify a loss type (uni, sum).",
    )

    parser.add_argument(
        "--gpu_id", type=int, default=0, help="0 for NAIS_prod, 1 for NAIS_concat"
    )

    parser.add_argument(
        "--node_dropout",
        nargs="?",
        default="[0.1]",
        help="Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.",
    )
    parser.add_argument(
        "--mess_dropout",
        nargs="?",
        default="[0.1,0.1,0.1]",
        help="Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.",
    )

    parser.add_argument("--K", type=int, default=10, help="Topk size")

    parser.add_argument(
        "--save_flag",
        type=int,
        default=1,
        help="0: Disable model saver, 1: Activate model saver",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="After how many epochs save ckpt"
    )

    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="Specify the test type from {part, full}, indicating whether the reference is done in mini-batch",
    )

    parser.add_argument(
        "--report",
        type=int,
        default=0,
        help="0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels",
    )

    parser.add_argument(
        "--use_att", type=bool, default=True, help="whether using attention mechanism"
    )
    parser.add_argument(
        "--use_kge",
        type=bool,
        default=True,
        help="whether using knowledge graph embedding",
    )
    parser.add_argument(
        "--print_every", type=int, default=20, help="Iter interval of printing loss."
    )
    parser.add_argument(
        "--l1_flag",
        type=bool,
        default=True,
        help="Flase: using the L2 norm, True: using the L1 norm.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether to log to wandb (requires setting the api key from command line as wandb login YOUR-API-KEY)",
    )
    parser.add_argument(
        "--wandb_entity",
        required="--wandb" in sys.argv,
        type=str,
        help="Entity name to push to the wandb logged data, in case args.wandb is specified.",
    )
    args = parser.parse_args()
    args.weight_dir = get_weight_dir("kgat", args.dataset)
    args.weight_dir_ckpt = get_weight_ckpt_dir("kgat", args.dataset)
    log_dir = os.path.join("logs", args.dataset, args.model_type)
    args.log_dir = log_dir
    return args
