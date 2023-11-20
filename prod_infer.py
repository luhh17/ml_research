import os
from utils.logger import print_argparse
import torch
from exp.base_exp import BaseExp
from exp.sklearn_exp import SklearnExp
from exp.increm_exp import IncremExp
import pickle as pkl
import pandas as pd
import numpy as np
import pdb
import yaml
from prepare_dataset.filter_and_construct import check_incremental_data_exist
import sys
from utils.logger import build_model_file_path_from_config
from agg_result import ModelEnsemble


infer_args = yaml.safe_load(open('./infer_config.yml', 'r'))
infer_args['devices'] = infer_args['devices'].replace(' ', '')
os.environ["CUDA_VISIBLE_DEVICES"] = infer_args['devices']
import torch
assert torch.cuda.is_available()




# '''
# rolling训练模型
# '''
def rolling_model() -> None:
    for config in infer_args['config_list']:
        args = yaml.safe_load(open(config, 'r'))
        if 'hidden_dim' in args:
            if args['hidden_dim'] == -1:
                args['hidden_dim'] = args['embed_dim'] * 4
        args['devices'] = infer_args['devices']
        args['epochs'] = infer_args['epochs']
        args['lr'] = infer_args['lr']
        if args['model'] == 'mlp':
            args['window_length'] = 1
        args['device'] = torch.device('cuda')
        torch.set_num_threads(6)
        check_incremental_data_exist(args, infer_args['cur_date'])

        args['task_name'], model_file_path = build_model_file_path_from_config(args)
        for model_idx in range(args['ensemble_num']):
            args['model_file_path'] = os.path.join(model_file_path, str(model_idx))
            print_argparse(args)
            if args['model'] in ['mlp', 'lstm', 'rnn', 'transformer', 'targetformer']:
                exp = IncremExp(args, year)
            elif args['model'] in ['xgb']:
                raise ValueError('xgb is not supported in incremental training')
            exp.save_args()
            exp.train()
            torch.cuda.empty_cache()
    


if __name__ == '__main__':
    rolling_model()
    ensemble = ModelEnsemble(suffix=)
    ensemble.ensemble_via_ols()
