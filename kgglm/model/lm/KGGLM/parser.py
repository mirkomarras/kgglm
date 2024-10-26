import argparse


def parser_kgglm_args():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument(
        "--task",
        type=str,
        default="pretrain",
        help="{pretrain, finetuneLP, finetuneRec" " end-to-end (originalPEARLM)}",
    )

    parser.add_argument(
        "--sample_size",
        type=str,
        default="500",
        help="Number of sampled path in the chosen dataset",
    )
    parser.add_argument(
        "--n_hop",
        type=int,
        default=5,
        help="Number of elements in a predicted sequence (considering only the ids)",
    )

    parser.add_argument(
        "--logit_processor_type",
        type=str,
        default="gcd",
        help="Path sequence deconding method: default to Graph Constrained Decoding",
    )
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Model to use from HuggingFace pretrained models",
    )
    parser.add_argument(
        "--nproc", type=int, default=8, help="Number of processes for dataset mapping"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Train batch size")
    parser.add_argument(
        "--test_batch_size", type=int, default=64, help="Test batch size"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=24,
        help="Context length value when training a tokenizer from scratch",
    )
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default="cuda:0", help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default="1", help="")
    parser.add_argument(
        "--infer_batch_size", type=int, default=64, help="Inference batch size"
    )
    parser.add_argument(
        "--n_seq_infer",
        type=int,
        default=30,
        help="Number of sequences generated for each user",
    )
    parser.add_argument(
        "--n_seq_infer_lp",
        type=int,
        default=30,
        help="Number of sequences generated for link prediction",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=50,
        help="Number of sequences generated for each user",
    )
    parser.add_argument(
        "--n_beams_lp",
        type=int,
        default=30,
        help="Number of sequences generated for link prediction",
    )

    # Parameter relative to resume training
    parser.add_argument(
        "--continue_training",
        type=bool,
        default=False,
        help="Whether to continue training from a checkpoint or not",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="Checkpoint from which to resume training of the model (default to starting from scratch)",
    )

    # Parameter relative to weight initialization
    parser.add_argument(
        "--logging_interval",
        type=int,
        default=100,
        help="Logging interval of the losses",
    )
    parser.add_argument(
        "--validation_interval", type=int, default=3000, help="Validation interval"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")

    args = parser.parse_args()

    return args
