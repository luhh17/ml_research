import sys
import os

class Logger(object):
    def __init__(self, file_path, also_print_to_terminal):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf8')
        self.print_to_terminal = also_print_to_terminal

    def write(self, message):
        if self.print_to_terminal:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def redirect_print_to_file(file_path, print_to_terminal=False):
    sys.stdout = Logger(file_path, print_to_terminal)


def print_argparse(args):
    for arg, value in args.items():
        print(f'{arg}: {value}')

def build_model_file_path_from_config(args: dict) -> str:
    task_name = f'{args["model"]}_{args["ret_type"]}_{args["loss_function"]}_win_{args["window_length"]}_pred_{args["target_window"]}' 
    if 'xgb' not in args['model']:
        task_name += f'_batchsize_{args["batch_size"]}' \
        f'_embedDim_{args["embed_dim"]}_encLayer_{args["num_enc_layers"]}' \
        f'_drop_{args["dropout"]}_norm_{args["norm_layer"]}_lr_{args["lr"]}_{args["activation"]}'
    else:
        task_name += f'_maxDepth_{args["max_depth"]}' \
        f'_n_{args["n_estimators"]}_maxLeaves_{args["max_leaves"]}' \
        f'_maxBin_{args["max_bin"]}_lr_{args["learning_rate"]}_gamma_{args["gamma"]}_regLambda{args["reg_lambda"]}_regAlpha_{args["reg_alpha"]}'
    model_file_path = os.path.join(args['model_file_root_path'], task_name)
    for model_idx in range(args['ensemble_num']):
        idx_path = os.path.join(model_file_path, str(model_idx))
        if not os.path.exists(idx_path):
            os.makedirs(idx_path)
    return task_name, model_file_path
