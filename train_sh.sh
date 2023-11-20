nohup python -u main.py ./configs/mlp/mlp_config_mse.yml >0.txt 2>&1 &
nohup python -u main.py ./configs/rnn/rnn_config_mse.yml >1.txt 2>&1 &
nohup python -u main.py ./configs/targetformer/targetformer_config.yml >2.txt 2>&1 &
  