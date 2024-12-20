import argparse
import json
import os
import shutil
import subprocess
from os import makedirs

from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from helper.models.rl.PGPR.pgpr_utils import (BEST_CFG_DIR, BEST_CFG_FILE_PATH,
                                              BEST_TEST_METRICS_FILE_PATH,
                                              CFG_FILE_PATH, LFM1M,
                                              OPTIM_HPARAMS_LAST_K,
                                              OPTIM_HPARAMS_METRIC,
                                              TEST_METRICS_FILE_PATH, TMP_DIR)
from helper.utils import get_model_dir

TRAIN_FILE_NAME = os.path.join(get_model_dir('PGPR','rl'),'train_agent.py')
TEST_FILE_NAME = os.path.join(get_model_dir('PGPR','rl'),'test_agent.py')

def load_metrics(filepath):
    if not os.path.exists(filepath):
        return None    
    with open(filepath) as f:
        metrics = json.load(f)
    return metrics

def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f)

def save_cfg(configuration, filepath):
    with open(filepath, 'w') as f:
        json.dump(configuration, f)

def metrics_average(metrics):
    avg_metrics = dict()
    for k, v in metrics.items():
        avg_metrics[k] = sum(v)/max(len(v),1)
    return avg_metrics

def save_best(best_metrics, test_metrics, grid):
    dataset_name = grid["dataset"]
    if best_metrics is None:
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )
        save_metrics(test_metrics, f'{BEST_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_CFG_FILE_PATH[dataset_name] }')
        return
    
    x = sum(test_metrics[OPTIM_HPARAMS_METRIC][-OPTIM_HPARAMS_LAST_K:])/OPTIM_HPARAMS_LAST_K
    best_x = sum(best_metrics[OPTIM_HPARAMS_METRIC][-OPTIM_HPARAMS_LAST_K:])/OPTIM_HPARAMS_LAST_K
    # if avg total reward is higher than current best
    if x > best_x :
        shutil.rmtree(BEST_CFG_DIR[dataset_name])
        shutil.copytree(TMP_DIR[dataset_name], BEST_CFG_DIR[dataset_name] )
        save_metrics(test_metrics, f'{BEST_TEST_METRICS_FILE_PATH[dataset_name]}')
        save_cfg(grid, f'{BEST_CFG_FILE_PATH[dataset_name] }')


def main(args):
    chosen_hyperparam_grid = {"act_dropout": [0], 
    "batch_size": [32], 
    "dataset": [args.dataset],#["lfm1m", "ml1m"], 
    "do_validation": [True], 
    "ent_weight":[ 0.001, 0.01], 
    "epochs": [50], 
    "gamma": [0.99],
    "hidden": [[512, 256], [128, 64]], 
    "lr": [0.0001], 
    "max_acts": [250], 
    "max_path_len": [3], 
    "name": ["train_agent"], 
    "seed": [123], 
    "state_history": [1]}

    makedirs(args.dataset)
    def prompt():
        answer = input("Continue (deletes content)? (y/n)")
        if answer.upper() in ["Y", "YES"]:
            return True
        elif answer.upper() in ["N", "NO"]:
            return False
    def can_run(dataset_name):
        if len(os.listdir(BEST_CFG_DIR[dataset_name])) > 0:
            print(f'Action required: WARNING {dataset_name} best hyper parameters folder is not empty')
            if not prompt():
                print('Content not deleted, To run grid search re-run the script and confirm deletion')
                return False
            else:
                shutil.rmtree(BEST_CFG_DIR[dataset_name])
                print('Content deleted\n Start grid search')
        return True
    for dataset_name in chosen_hyperparam_grid['dataset']:
        if not can_run(dataset_name):
            return 

    hparam_grids = ParameterGrid(chosen_hyperparam_grid)
    print('num_experiments: ', len(hparam_grids))

    for i, configuration in enumerate(tqdm(hparam_grids)):
        dataset_name = configuration["dataset"]
        makedirs(dataset_name)
            
        CMD = ["python3", TRAIN_FILE_NAME]

        for k,v in configuration.items():
                if isinstance(v,list):
                    cmd_args = [f'--{k}'] + [f" {val} " for val in v]
                    CMD.extend( cmd_args )
                else:
                    CMD.extend( [f'--{k}', f'{v}'] )  
        print(f'Executing job {i+1}/{len(hparam_grids)}: ',configuration)
        subprocess.call(CMD)
        save_cfg(configuration, CFG_FILE_PATH[dataset_name])        
        test_metrics = load_metrics(TEST_METRICS_FILE_PATH[dataset_name])
        best_metrics = load_metrics(BEST_TEST_METRICS_FILE_PATH[dataset_name])
        save_best(best_metrics, test_metrics, configuration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    args = parser.parse_args()
    main(args)