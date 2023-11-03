import os
import sys
import pickle

import torch
import numpy as np

from gcn.datasets.seeds import val_seeds
from gcn.utils.parser import parse_args
from gcn.utils.tools import get_filename_base
from gcn.utils.tools import load_graph_data
from gcn.datasets.tools import DATASET_DICT
from base import base

# to test reproducibility
np.random.seed(123)

def main(args=None, data_folder='data', output_folder='output'):

    if args is None:
        return

    if not args["no_cuda"]:
        if torch.cuda.device_count() > args["gpu_number"]:
            args["device"] = torch.device('cuda', args["gpu_number"])
            print(f'Setting up GPU {args["gpu_number"]}...')
        else:
            print(f'GPU {args["gpu_number"]} not available, setting up GPU 0...')
            args["device"] = torch.device('cuda', 0)
    else:
        print('GPU not available, setting up CPU...')
        args["device"] = torch.device('cpu')


    name_folder, init_name = DATASET_DICT.get(args["dataset"], [None, None])
    args["output_folder"] = output_folder

    # Search space general hyperparameters.
    lr_space = np.array([0.005, 0.02])
    weight_decay_space = np.array([1e-4, 1e-3])
    hidden_units_space = np.array([16, 32, 64])
    dropout_space = np.array([0.3, 0.7])
    n_layers_space = np.array([2, 3, 4, 5])

    # Search space hyperparameters of Jost & Liu curvature.
    if args["GNN"] in ['SSobGNN', 'SobGNN']:
        alpha_space = np.array([1, 2, 3, 4, 5, 6])
        epsilon_space = np.array([0.5, 2])
        aggregation_space = np.array(['linear', 'concat'])
    if args["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
        heads_space = np.array([1, 2, 3, 4, 5, 6])
    if args["GNN"] == 'Cheby':
        K_Cheby_space = np.array([1, 2, 3])
    if args["GNN"] == 'SIGN':
        K_SIGN_space = np.array([1, 2, 3])

    path_hyperparameters = os.path.join(output_folder, f'{args["GNN"]}_hyperTuning', args["graph"])

    # Check if there are work in this dataset
    if os.path.exists(path_hyperparameters):
        with open(os.path.join(path_hyperparameters, f'{args["dataset"]}.pkl'), 'rb') as f:
            hyperparameters = pickle.load(f)
        indx_zeros = np.where(hyperparameters["lr"] == 0)[0]
    else:
        os.makedirs(path_hyperparameters, exist_ok=True)
        hyperparameters = dict()
        hyperparameters["lr"] = np.zeros((args["iterations"],))
        hyperparameters["weight_decay"] = np.zeros((args["iterations"],))
        hyperparameters["hidden_units"] = np.zeros((args["iterations"],), dtype=int)
        hyperparameters["dropout"] = np.zeros((args["iterations"],))
        hyperparameters["n_layers"] = np.zeros((args["iterations"],), dtype=int)
        if args["GNN"] in ['SSobGNN', 'SobGNN']:
            hyperparameters["alpha"] = np.zeros((args["iterations"],), dtype=int)
            hyperparameters["epsilon"] = np.zeros((args["iterations"],))
            hyperparameters["aggregation"] = np.zeros((args["iterations"],), dtype=object)#dict()
        if args["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
            hyperparameters["heads"] = np.zeros((args["iterations"],))
        if args["GNN"] == 'Cheby':
            hyperparameters["K_Cheby"] = np.zeros((args["iterations"],), dtype=int)
        if args["GNN"] == 'SIGN':
            hyperparameters["K_SIGN"] = np.zeros((args["iterations"],), dtype=int)
        indx_zeros = np.where(hyperparameters["lr"] == 0)[0]

    experiments = []
    for i in range(0, args["iterations"]):
        sys.stdout.flush()
        repeat_flag = False
        if i not in indx_zeros:
            params = {key : value[i] for key, value in hyperparameters.items()}
            params.update({"dataset" : args["dataset"], "epochs" : args["epochs"], "GNN" : args["GNN"]})
            file_name_base = get_filename_base(params["n_layers"], params)
            file_name = os.path.join(output_folder, f'{args["GNN"]}_hyperTuning', args["graph"], f'{file_name_base}.pkl')
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    best_acc_test_vec, _ = pickle.load(f)
                indx_zero = np.where(best_acc_test_vec == 0)
                if indx_zero[0].size > 0:
                    repeat_flag = True
                    print(f'Repeating experiment {i}.')
                    sys.stdout.flush()
                else:
                    print(f'Experiment {i} is fine.')
            else:
                repeat_flag = True
                print('Repeating experiment.')
                sys.stdout.flush()
        if (i in indx_zeros) or repeat_flag:
            experiments.append(i)
            # Random sampling in the search space.
            lr = np.random.uniform(lr_space[0], lr_space[1], size=1)
            lr = np.round(lr, decimals=4)
            hyperparameters["lr"][i] = lr
            weight_decay = np.random.uniform(weight_decay_space[0], weight_decay_space[1], size=1)
            weight_decay = np.round(weight_decay, decimals=4)
            hyperparameters["weight_decay"][i] = weight_decay
            hidden_units = np.random.choice(hidden_units_space, size=1)
            hyperparameters["hidden_units"][i] = hidden_units
            dropout = np.random.uniform(dropout_space[0], dropout_space[1], size=1)
            dropout = np.round(dropout, decimals=4)
            hyperparameters["dropout"][i] = dropout
            n_layers = np.random.choice(n_layers_space, size=1)
            hyperparameters["n_layers"][i] = n_layers
            if args["GNN"] in ['SSobGNN', 'SobGNN']:
                alpha = np.random.choice(alpha_space, size=1)
                hyperparameters["alpha"][i] = alpha
                epsilon = np.random.uniform(epsilon_space[0], epsilon_space[1], size=1)
                epsilon = np.round(epsilon, decimals=4)
                hyperparameters["epsilon"][i] = epsilon
                aggregation = np.random.choice(aggregation_space, size=1)
                hyperparameters["aggregation"][i] = aggregation[0]
            if args["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
                heads = np.random.choice(heads_space, size=1)
                hyperparameters["heads"][i] = heads
            if args["GNN"] == 'Cheby':
                K_Cheby = np.random.choice(K_Cheby_space, size=1)
                hyperparameters["K_Cheby"][i] = K_Cheby
            if args["GNN"] == 'SIGN':
                K_SIGN = np.random.choice(K_SIGN_space, size=1)
                hyperparameters["K_SIGN"][i] = K_SIGN

            file_name = os.path.join(output_folder, f'{args["GNN"]}_hyperTuning', args["graph"], f'{args["dataset"]}.pkl')
            with open(file_name, 'wb') as f:
                pickle.dump(hyperparameters, f)

    for i in experiments:
        print('Search iteration: ' + str(i))
        params = {key : value[i] for key, value in hyperparameters.items()}
        args.update(params)
        data, n = load_graph_data(args, root_folder=data_folder, name_folder=name_folder, init_name=init_name)
        base(data, n, args, seeds=val_seeds)

if __name__ == '__main__':
    args = parse_args(train_mode=False)
    main(args, data_folder='data', output_folder='results_hyper_3')