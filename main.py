import os
from utils.logger import print_argparse
import torch
from exp.base_exp import BaseExp
from exp.sklearn_exp import SklearnExp
import pickle as pkl
import pandas as pd
import numpy as np
import pdb
import yaml
from prepare_dataset.filter_and_construct import check_data_exist
import sys

from utils.logger import build_model_file_path_from_config

args = yaml.safe_load(open(sys.argv[1], 'r'))
if 'hidden_dim' in args:
    if args['hidden_dim'] == -1:
        args['hidden_dim'] = args['embed_dim'] * 4
args['devices'] = args['devices'].replace(' ', '')
os.environ["CUDA_VISIBLE_DEVICES"] = args['devices']
import torch
assert torch.cuda.is_available()
if args['model'] == 'mlp':
    args['window_length'] = 1
args['device'] = torch.device('cuda')
torch.set_num_threads(6)
check_data_exist(args, args['test_year_list'])



# '''
# rolling训练模型
# '''
def rolling_model() -> None:
    if not os.path.exists(args['model_file_root_path']):
        os.makedirs(args['model_file_root_path'])
    args['task_name'], model_file_path = build_model_file_path_from_config(args)
    for model_idx in range(args['ensemble_num']):
        args['model_file_path'] = os.path.join(model_file_path, str(model_idx))
        for year in args['test_year_list']:
            if os.path.exists(os.path.join(args['model_file_path'], f'{year}_pred_result')):
                w = pkl.load(open(os.path.join(args['model_file_path'], f'{year}_pred_result'), 'rb'))
                if len(w) > 0:
                    continue
            print_argparse(args)
            if args['model'] in ['mlp', 'lstm', 'rnn', 'transformer', 'targetformer']:
                exp = BaseExp(args, year)
            elif args['model'] in ['xgb']:
                exp = SklearnExp(args, year)
            exp.save_args()
            exp.train()
            torch.cuda.empty_cache()
       


if __name__ == '__main__':
    rolling_model()
