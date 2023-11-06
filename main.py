import os
from utils.logger import redirect_print_to_file, print_argparse
import torch
import argparse
from exp.base_exp import Base_Exp
from exp.pretrain_exp import PretrainExp
from exp.post_exp import Post_Exp
import pickle as pkl
from prepare_dataset.myDataset import RetDataset, divide_train_valid_test_ff3
from prepare_dataset.myDataset import RetDataset, BalancedDataset
import pandas as pd
import numpy as np
import pdb

parser = argparse.ArgumentParser(description='Deep Learning in Portfolio Construction')
parser.add_argument('--model', type=str, default='mlp',
                    help='model of experiment, options: [informer, transformer, mlp, rnn, targetformer]')

parser.add_argument('--devices', type=str, default='3', help='device ids of multile gpus, can be like 0,1,2,3')
parser.add_argument('--mlp_implement', type=str, default='linear', help='mlp implement, options: [linear, conv]')
parser.add_argument('--mlp_type', type=str, default='res', help='mlp type, options: [vanilla, res]')

parser.add_argument('--mktcap_filter', type=int, default=2000, choices=[2000, -1], help='select top k mktcap stocks')
parser.add_argument('--norm_method', type=str, default='std', choices=['rank', 'std'], help='normalization method')
parser.add_argument('--test_start_date', type=str, default='2017-01-01', help='test start date')
parser.add_argument('--test_end_date', type=str, default='2021-12-31', help='test end date')
parser.add_argument('--train_start_date', type=str, default='2005-01-01', help='train start date')
parser.add_argument('--train_end_date', type=str, default='2015-12-31', help='train end date')
parser.add_argument('--valid_start_date', type=str, default='2016-01-01', help='valid start date')
parser.add_argument('--valid_end_date', type=str, default='2016-12-31', help='valid end date')
parser.add_argument('--est_method', type=str, default='rolling', choices=['single', 'rolling'], help='single model or rolling model')
parser.add_argument('--train_data_range', type=str, default='roll', choices=['expand', 'roll'], help='single model or rolling model')

parser.add_argument('--capacity', type=int, default=512, help='capacity of replay buffer')

parser.add_argument('--ret_type', type=str, default='ret_open', choices=['ret_open', 'exret_open','ret_open_3d','ret_open_5d','ret_open_10d','exret_open_3d','exret_open_5d','exret_open_10d'],help='which ret to use')
parser.add_argument('--loss_function', type=str, default='MSE', choices=['MSE', 'Sharpe','MSEIC'],help='which loss function to use')
parser.add_argument('--mode', type=str, default='basic',  choices=['basic', 'intraday_only','totalfea','dualnet'], help='mode of experiment')


parser.add_argument('--match_stkcd_date', action="store_true", default=False, help='whether load stkcd and date matrix')
parser.add_argument('--topk', type=int, default=200, help='top k stocks to select in one direction')
parser.add_argument('--norm', type=str, default='raw', help='normalization method on time series, options: [raw, min_max, mean_var]')
parser.add_argument('--mask_ratio', type=float, default=0.2)
parser.add_argument('--optim', type=str, default='adam', help='optimizer of experiment, options: [adam, sgd]')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--epochs', type=int, default=150, help='train epochs')
parser.add_argument('--train_strategy', type=str, default='earlyStopping', choices=['full', 'earlyStopping'], help='train full epochs or early stopping')
parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'exponential'], help='learning rate scheduler')
parser.add_argument('--sample_policy', type=str, default='random', choices=['fixed', 'random'], help='sample policy')
parser.add_argument('--gradient_std', type=str, default='std', choices=['std', 'notStd'], help='whether to divide loss by batch_size for gradient accumulation')
parser.add_argument('-use_amp', action="store_true", default=False, help='use automatic mixed precision')
parser.add_argument('--window_length', type=int, default=5, help='window size (num of days) of input data')
parser.add_argument('--max_steps', type=int, default=20, help='holding length (num of days) for evaluating SR and MDD in portfolio optimization')
parser.add_argument('--pred_length', type=int, default=1, help='prediction length')
parser.add_argument('--dec_len_from_input', type=int, default=5, help='use the last len input as the decoder input')
parser.add_argument('--target', type=str, default='sharpe')
parser.add_argument('--gamma', type=float, default=1, help='discount factor')
parser.add_argument('--sample_factor', type=float, default=0.1, help='sample factor')
parser.add_argument('--weight_adj', type=str, default='self', help='whether to adjust prob as weight or use raw prob')

parser.add_argument('--embed_dim', type=int, default=128, help='embedding layer dimension of encoder')
parser.add_argument('--dec_embed_dim', type=int, default=64, help='embedding layer dimension of decoder')

parser.add_argument('--char_dim', type=int, default=1771, help='number of character')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden state dimension of lstm / hidden state dimension of ffn in transformer')
parser.add_argument('--num_enc_layers', type=int, default=2, help='number of encoder layers')
parser.add_argument('--num_dec_layers', type=int, default=1, help='number of decoder layers')
parser.add_argument('--num_heads', type=int, default=4, help='number of heads in multi-head attention')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--atten', type=str, default='full', choices=['prob', 'full', 'causal', 'compress_causal', 'target', 'compress_target'], help='attention used in transformer')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'], help='activation function')
parser.add_argument('-distil', action="store_true", default=False, help='whether to use distilling in transformer')
parser.add_argument('-use_decoder',  action="store_true", default=False, help='whether to use decoder in transformer')
parser.add_argument('-baseline', action='store_true', default=False, help='whether use baseline in update')
parser.add_argument('-explore_noise', action='store_true', default=False, help='whether add noise for exploration')
parser.add_argument('--norm_layer', type=str, default='batchNorm', choices=['batchNorm', 'layerNorm'])
parser.add_argument('--model_file_path', type=str, default='../model_file', help='model file path, save checkpoint attention config all in one')
parser.add_argument('--task_name', type=str, default='rnn_window_20', help='task name')
parser.add_argument('-output_attention', action="store_true", default=False, help='whether to output attention')
parser.add_argument('--mask_target', type=str, default='signal', choices=['signal', 'ret'], help='mask target')
parser.add_argument('-cat_ret', action="store_true", default=False, help='whether to concatenate return to input')
parser.add_argument('-target_self', action="store_true", default=False, help='whether to calculate target self attention')
parser.add_argument('-subsample', action="store_true", default=False, help='whether to use subsample of data')
parser.add_argument('-reverse', action="store_true", default=False, help='whether to reverse the values')
parser.add_argument('-only_one_target', action="store_true", default=False, help='whether to insert only one target attention')
parser.add_argument('-target_first', action="store_true", default=False, help='whether to calculate target-self-attention first')
parser.add_argument('-more_self', action="store_true", default=False, help='whether to insert x-self attention in compress attention')
parser.add_argument('--last_atten', type=str, default='mean', choices=['mean', 'last'], help='how to deal with last attention, average over all valus or take the earliest one')
parser.add_argument('-contemporary', action="store_true", default=False, help='whether to mask the diagnal of attention')
parser.add_argument('-temp_embed', action="store_true", default=False, help='whether to add temporal embedding')
parser.add_argument('--embed_type', type=str, default='timeF', choices=['timeF', 'fixed', 'else'], help='temporal embedding type')
parser.add_argument('--factor', type=float, default=0.7,  help='factor for logk complexity')
parser.add_argument('--label_len', type=int, default=5,  help='label length for decoder init')
parser.add_argument('--pred_len', type=int, default=1,  help='prediction length')
parser.add_argument('--moving_avg', type=int, default=5,  help='moving average window length')
parser.add_argument('-prenorm', action="store_true", default=False, help='whether to use pre-norm in transformer')

args = parser.parse_args()
if args.hidden_dim == -1:
    args.hidden_dim = args.embed_dim * 4
if args.cat_ret:
    args.char_dim += 1
args.devices = args.devices.replace(' ', '')
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
import torch
assert torch.cuda.is_available()
if args.model == 'mlp':
    args.window_length = 1
args.device = torch.device('cuda')
torch.set_num_threads(6)
if not args.use_decoder:
    args.num_dec_layers = 0


def get_data_year(args):
    """
    Load and preprocess data for a given test year.

    Args:
        args: Namespace object containing command-line arguments.

    Returns:
        Tuple of PyTorch DataLoader objects for training, validation, and testing datasets.
    """

    test_year = args.test_start_date.split('-')[0]
    print(f'loading data {test_year}')

    if args.mode=='basic':
        train, valid, test, basic_fea_num= pkl.load(open(f'/mnt/HDD16TB/data_output_20231015/{args.train_data_range}_{test_year}_{args.ret_type}_{args.window_length}_balanced.pkl', 'rb'))
        args.char_dim=basic_fea_num
        
    train_dataset = BalancedDataset(train, args)
    valid_dataset = BalancedDataset(valid, args)    
    test_dataset = BalancedDataset(test, args)
    
    del train,valid,test

    if args.loss_function=='MSEIC':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.loss_function=='Sharpe':
        # don't shuffle the samples to ensure the economic meaning of sharpe ratio
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    del train_dataset,valid_dataset,test_dataset

    print('loading data done!')
    return (train_loader,valid_loader,test_loader)

'''
rolling训练模型
'''
def rolling_model():
    for year in [2018, 2019,2020,2021,2022,2023]:
        args.train_start_date = f'{year - 3}-01-01'
        args.train_end_date = f'{year - 1}-08-31'
        args.valid_start_date = f'{year - 1}-09-01'
        args.valid_end_date = f'{year - 1}-12-31'
        args.test_start_date = f'{year}-01-01'
        args.test_end_date = f'{year}-12-31'
        
        finish_loading=False
        for model_idx in range(5):
            args.model_file_path = f'./../ensemble_model_file_{args.mode}'
            if args.model == 'mlp':
                args.task_name = f'{args.model}_{args.mlp_type}_{args.ret_type}_loss_{args.loss_function}' \
                                f'_pred_{args.pred_length}_batchsize_{args.batch_size}' \
                                f'_embedDim_{args.embed_dim}_encLayer_{args.num_enc_layers}' \
                                f'_drop_{args.dropout}'
            elif args.model == 'lstm':
                args.task_name = f'{args.model}_{args.ret_type}_loss_{args.loss_function}' \
                                f'_pred_{args.pred_length}_batchsize_{args.batch_size}' \
                                f'_embedDim_{args.embed_dim}_encLayer_{args.num_enc_layers}' \
                                f'_drop_{args.dropout}'
            elif args.model == 'autoformer':
                args.task_name = f'{args.model}_{args.ret_type}_loss_{args.loss_function}_win_{args.window_length}'\
                                f'_pred_{args.pred_length}_mvavg_{args.moving_avg}' \
                                f'_embedDim_{args.embed_dim}_encLayer_{args.num_enc_layers}_useDec_{args.use_decoder}' \
                                    f'_decLayer_{args.num_dec_layers}_head_{args.num_heads}_drop_{args.dropout}'
            elif args.model == 'targetformer':
                args.task_name = f'{args.model}_{args.ret_type}_loss_{args.loss_function}_win_{args.window_length}_prenorm_{args.prenorm}_targetSelf_{args.target_self}_' \
                                f'catRet_{args.cat_ret}_oneTarget_{args.only_one_target}_targetFirst_{args.target_first}' \
                                f'_pred_{args.pred_length}' \
                                f'_embedDim_{args.embed_dim}_encLayer_{args.num_enc_layers}_decLayer_{args.num_dec_layers}' \
                                    f'_decLen_{args.dec_len_from_input}_head_{args.num_heads}_drop_{args.dropout}'
            else:
                args.task_name = f'{args.model}_{args.ret_type}_loss_{args.loss_function}_win_{args.window_length}_atten_{args.atten}_useDecoder_{args.use_decoder}_' \
                                f'decLen_{args.dec_len_from_input}' \
                                f'_pred_{args.pred_length}' \
                                f'_embedDim_{args.embed_dim}_encLayer_{args.num_enc_layers}_decLayer_{args.num_dec_layers}' \
                                    f'_head_{args.num_heads}_drop_{args.dropout}'
                
            args.model_file_path = os.path.join(args.model_file_path, args.task_name)
            args.model_file_path = os.path.join(args.model_file_path, str(model_idx))
            if not os.path.exists(args.model_file_path):
                os.makedirs(args.model_file_path)
            if os.path.exists(os.path.join(args.model_file_path, f'{year}_pred_result')):
                w = pkl.load(open(os.path.join(args.model_file_path, f'{year}_pred_result'), 'rb'))
                if len(w) > 0:
                    continue
            redirect_print_to_file(os.path.join(args.model_file_path, f"{args.test_start_date.split('-')[0]}_log.txt"), print_to_terminal=True)
            print_argparse(args)

            if finish_loading==False:
                exp_loader=get_data_year(args)
                finish_loading=True

            exp = Base_Exp(args,exp_loader)
            exp.save_args()
            exp.train()
            exp.test_model()
            torch.cuda.empty_cache()
        
        if finish_loading==True:
            del exp_loader



if __name__ == '__main__':
    rolling_model()
