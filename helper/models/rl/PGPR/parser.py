import argparse
import torch
from helper.knowledge_graphs.kg_macros import  ML1M

def parser_pgpr_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default=ML1M, help='One of {ML1M}')
    parser.add_argument('--name', type=str,
                        default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250,
                        help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int,
                        default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float,
                        default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float,
                        default=0, help='action dropout rate.')
    parser.add_argument('--state_history', type=int,
                        default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*',
                        default=[512, 256], help='number of samples')
    parser.add_argument('--do_validation', type=bool,
                        default=True, help='Whether to perform validation')
    args = parser.parse_args()

    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    
    return args


def parser_pgpr_test():
    def boolean(x): return (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M,
                        help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str,
                        default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=1, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250,
                        help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int,
                        default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='reward discount factor.')
    parser.add_argument('--state_history', type=int,
                        default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*',
                        default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean,
                        default=True, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=list, nargs='*',
                        default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True,
                        help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean,
                        default=True, help='Run evaluation?')
    parser.add_argument('--save_paths', type=boolean,
                        default=False, help='Save paths')
    
    # For Embeddings
    parser.add_argument('--embed_name', type=str,
                        default='TransE', help='Which kge embedding model?')
    args = parser.parse_args()

    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    return args
