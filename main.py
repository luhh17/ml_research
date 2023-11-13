import os
from utils.logger import print_argparse
import torch
from exp.base_exp import BaseExp
import pickle as pkl
import pandas as pd
import numpy as np
import pdb
import yaml
from prepare_dataset.filter_and_construct import check_data_exist


with open('./config.yml') as f:
    args = yaml.safe_load(f)
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
    for model_idx in range(args['ensemble_num']):
        for year in args['test_year_list']:
            if not os.path.exists(args['model_file_path']):
                os.makedirs(args['model_file_path'])

            args['task_name'] = f'{args["model"]}_{args["ret_type"]}_{args["loss_function"]}' \
            f'_win_{args["window_length"]}_pred_{args["pred_length"]}' 
            
            if 'prod' not in args['mode']:
                args['task_name'] += f'_batchsize_{args["batch_size"]}' \
                f'_embedDim_{args["embed_dim"]}_encLayer_{args["num_enc_layers"]}' \
                f'_drop_{args["dropout"]}_norm_{args["norm_layer"]}_lr_{args["lr"]}_{args["activation"]}' \
    
            args['model_file_path'] = os.path.join(args['model_file_path'], args['task_name'])
            args['model_file_path'] = os.path.join(args['model_file_path'], str(model_idx))
            if not os.path.exists(args['model_file_path']):
                os.makedirs(args['model_file_path'])
            if args['mode'] == 'train':
                if os.path.exists(os.path.join(args['model_file_path'], f'{year}_pred_result')):
                    w = pkl.load(open(os.path.join(args['model_file_path'], f'{year}_pred_result'), 'rb'))
                    if len(w) > 0:
                        continue
                print_argparse(args)
                if args['model'] in ['mlp', 'lstm', 'rnn', 'transformer', 'targetformer']:
                    exp = BaseExp(args, year)
                exp.save_args()
                exp.train()
                torch.cuda.empty_cache()
            elif args['mode'] == 'infer':
                if args['model'] in ['mlp', 'lstm', 'rnn', 'transformer', 'targetformer']:
                    exp = BaseExp(args, year)
                exp.save_args()
                exp.inference()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    rolling_model()
