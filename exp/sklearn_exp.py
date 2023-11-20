import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN, LSTM
from models.transformer.transformer import Transformer
from models.targetformer.targetformer import Targetformer
from models.ffn import Mlp3D, Mlp2D, MLP_linear_concat
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import BalancedDataset
from utils.loss_func import sharpe_loss, mseic_loss
from exp.base_exp import BaseExp
import pdb
import cupy as cp
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class SklearnExp(BaseExp):
    def __init__(self, args: dict[str, int], test_year: int) -> None:
        super(SklearnExp, self).__init__(args, test_year)
        self.args = args
        self.device = self._acquire_device()
        self.test_year = test_year
        self._get_data()
        self.model = self._build_model()
        self.save_path = f"{self.args['model_file_path']}/{test_year}_checkpoint.json"


    def save_args(self) -> None:
        pkl.dump(self.args, open(f"{self.args['model_file_path']}/config", 'wb'))

    def _build_model(self) -> nn.Module:
        if self.args['model'] == 'xgb':
            model = xgb.XGBRegressor(n_jobs=1, tree_method="hist", early_stopping_rounds=self.args['patience'], max_depth=self.args['max_depth'], 
            n_estimators=self.args['n_estimators'], max_leaves=self.args['max_leaves'], max_bin=self.args['max_bin'], learning_rate=self.args['learning_rate'], 
            gamma=self.args['gamma'], reg_lambda=self.args['reg_lambda'], reg_alpha=self.args['reg_alpha'])
        return model

    def _get_data(self):
        train, valid, test, basic_fea_num = pkl.load(open(f"{self.args['output_dataset_path']}/roll_{self.test_year}_{self.args['ret_type']}_{self.args['target_window']}d_{self.args['window_length']}_balanced.pkl", 'rb'))
        X = train[0].reshape(-1, basic_fea_num)
        y = train[1].reshape(-1, 1)
        mask = train[3].reshape(-1, 1)
        X_valid = valid[0].reshape(-1, basic_fea_num)
        y_valid = valid[1].reshape(-1, 1)
        mask_valid = valid[3].reshape(-1, 1)
        X_test = test[0].reshape(-1, basic_fea_num)
        y_test = test[1].reshape(-1, 1)
        mask_test = test[3].reshape(-1, 1)
        date_test = test[4].reshape(-1, 1)
        stkcd_test = test[5].reshape(-1, 1)

        
        X = X[mask.reshape(-1)==1]
        y = y[mask.reshape(-1)==1]
        X_valid = X_valid[mask_valid.reshape(-1)==1]
        y_valid = y_valid[mask_valid.reshape(-1)==1]
        X_test = X_test[mask_test.reshape(-1)==1]
        y_test = y_test[mask_test.reshape(-1)==1]
        date_test = date_test[mask_test.reshape(-1)==1]
        stkcd_test = stkcd_test[mask_test.reshape(-1)==1]

        self.train_data = (X, y)
        self.valid_data = (X_valid, y_valid)
        self.test_data = (X_test, y_test, date_test, stkcd_test)


    def train(self) -> None:
        self.model.fit(self.train_data[0], self.train_data[1], eval_set=[self.valid_data])
        self.model.save_model(self.save_path)
        pred = self.model.predict(self.test_data[0])
        pred_df = pd.DataFrame({'date': self.test_data[2].reshape(-1), 'stkcd': self.test_data[3].reshape(-1), 'pred': pred.reshape(-1)})
        pkl.dump(pred_df, open(f"{self.args['model_file_path']}/{self.test_year}_pred_result", 'wb'))
        return


    def hyper_tune(self) -> nn.Module:
        train, valid, test, basic_fea_num = pkl.load(open(f"{self.args['output_dataset_path']}/roll_{self.test_year}_{self.args['ret_type']}_{self.args['target_window']}d_{self.args['window_length']}_balanced.pkl", 'rb'))
        X = train[0].reshape(-1, basic_fea_num)
        y = train[1].reshape(-1, 1)
        mask = train[3].reshape(-1, 1)
        X_test = test[0].reshape(-1, basic_fea_num)
        y_test = test[1].reshape(-1, 1)
        mask_test = test[3].reshape(-1, 1)
        
        X = X[mask.reshape(-1)==1]
        y = y[mask.reshape(-1)==1]
        X_test = X_test[mask_test.reshape(-1)==1]
        y_test = y_test[mask_test.reshape(-1)==1]
        
        # exit()
        xgb_model = xgb.XGBRegressor(n_jobs=1, tree_method="hist", 
        early_stopping_rounds=self.args['patience'], device="cuda")
        clf = GridSearchCV(xgb_model,
        {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200], "max_leaves": [0, 100], "max_bin": [32, 256], "learning_rate": [1e-1, 1e-2], 
        "gamma": [0, 1], "reg_lambda": [0, 1], "reg_alpha": [0, 1]}, 
        verbose=1, n_jobs=2,)
        clf.fit(X, y, eval_set=[(X_test, y_test)])
        print(clf.best_params)
        gpu_res = clf.evals_result()
        print(gpu_res)
        clf.save_model(self.save_path)
        return

