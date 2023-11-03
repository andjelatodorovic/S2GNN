import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from scipy.io import loadmat

from gcn.nets import (SSobGNN, GCN, Cheby, kGNN, GAT,
                    Transformer, SGC, ClusterGCN, FiLM,
                    SuperGAT, GATv2, ARMA, SIGN)

from gcn.datasets import PPRDataset, get_dataset


def save_results(results, args, n_layers):
    file_name_base = get_filename_base(n_layers, args)
    if args["hyperparameterTunning_mode"]:
        path = os.path.join(args["output_folder"], f'{args["GNN"]}_hyperTuning', args["graph"])
    else:
        path = os.path.join(args["output_folder"], args["GNN"], args["graph"])
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if args["hyperparameterTunning_mode"]:
        with open(os.path.join(path, f'{file_name_base}.pkl'), 'wb') as f:
            pickle.dump(results['hyper'], f)
    else:
        with open(os.path.join(path, f'{file_name_base}.pkl'), 'wb') as f:
            pickle.dump(results['train'], f)


def get_filename_base(n_layers, args):

    filename_base = f'{args["dataset"]}_nL_{n_layers}_hU_{args["hidden_units"]}_lr_' + \
                         f'{args["lr"]}_wD_{args["weight_decay"]}_dr_{args["dropout"]}_mE_{args["epochs"]}'
    if args["GNN"] in ['SSobGNN', 'SobGNN']:
        complement_filename = f'_alpha_{args["alpha"]}_epsilon_{args["epsilon"]}_aggregation_{args["aggregation"]}'
        filename_base = filename_base + complement_filename
    if args["GNN"] == 'Cheby':
        complement_filename = f'_KCheby_{args["K_Cheby"]}'
        filename_base = filename_base + complement_filename
    if args["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
        complement_filename = f'_headsAttention_{args["headsAttention"]}'
        filename_base = filename_base + complement_filename
        
    if args["GNN"] == 'SIGN':
        complement_filename = f'_KSIGN_{args["K_SIGN"]}'
        filename_base = filename_base + complement_filename

    return filename_base


def get_graph_data(points, labels, num_classes, adj, args):

    edge_index = []
    edge_attr = []

    if args["GNN"] == 'SSobGNN':
        adj_tilde = adj + args["epsilon"] * torch.eye(adj.shape[0])
        tensor_ones = torch.ones(adj.shape[0], 1)

        for rho in range(1, args["alpha"]+1):
            sparse_sobolev_term = torch.pow(adj_tilde, rho)
            edge_index_temp = sparse_sobolev_term.nonzero().T
            edge_index.append(edge_index_temp)
            edge_attr.append(sparse_sobolev_term[edge_index_temp[0, :], edge_index_temp[1, :]])
        data = Data(x=points, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    elif args["GNN"] == 'SobGNN':
        adj_tilde = adj + args["epsilon"] * torch.eye(adj.shape[0])
        tensor_ones = torch.ones(adj.shape[0], 1)
        for rho in range(1, args["alpha"]+1):
            sobolev_term = torch.matrix_power(adj_tilde, rho)
            edge_index_temp = sobolev_term.nonzero().T
            edge_index.append(edge_index_temp)
            edge_attr.append(sobolev_term[edge_index_temp[0, :], edge_index_temp[1, :]])
        data = Data(x=points, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    elif args["GNN"] == 'SIGN':
        xs = []
        tensor_ones = torch.ones(adj.shape[0], 1)
        diag_elements = torch.mm(adj, tensor_ones)
        diag_elements = torch.div(tensor_ones, torch.sqrt(diag_elements))
        diag_tilde = torch.diag(diag_elements.view(-1))
        filtering_function = torch.spmm(diag_tilde, torch.spmm(adj, diag_tilde))

        for k_sign in range(1, args["K_SIGN"] + 1):
            higher_term = torch.matrix_power(filtering_function, k_sign)
            xs.append(torch.mm(higher_term, points))

        data = Data(x=points, y=labels)
        data.xs = xs

    elif args["GNN"] in ['GCN', 'Cheby', 'kGNN', 'GAT', 'Transformer', 'SGC',
                      'ClusterGCN', 'FiLM', 'SuperGAT', 'GATv2', 'ARMA']:
        edge_index = adj.nonzero().T
        edge_attr = adj[edge_index[0, :], edge_index[1, :]]
        data = Data(x=points, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    data.num_features = points.shape[1]
    data.num_classes = num_classes

    return data


def load_graph_data(args, root_folder='data', name_folder=None, init_name=None):

    if name_folder:
        print(f'Loading {args["graph"]} graph')
        matfile = loadmat(os.path.join(root_folder, name_folder, f'{init_name}_graph_{args["graph"]}.mat'))

        # extract data from matfile
        adj_matrices = matfile['adj_matrices']
        points = matfile['points']
        labels = matfile['label_bin']
        num_classes = labels.shape[1]
        labels = torch.LongTensor(np.where(labels)[1])
        A = torch.FloatTensor(adj_matrices[0][0])
        points = torch.FloatTensor(points[0][0])

        data = get_graph_data(points, labels, num_classes, A, args)
        n = A.shape[0]

        return data, n

    else:
        # Load and preprocess data
        if args["GraphDifussion"]:
            dataset = PPRDataset(
                name=args["dataset"],
                use_lcc=True,
                alpha=args["alphaGDC"],
                k=args["k"]
            )
            if args["undirected"]:
                dataset._data.edge_index, dataset._data.edge_attr = to_undirected(dataset._data.edge_index, dataset._data.edge_attr)
            dataset._data.edge_index, dataset._data.edge_attr = remove_self_loops(dataset._data.edge_index, dataset._data.edge_attr)
        else:
            if args["dataset"] in ['chameleon', 'squirrel', 'Actor']:
                file_name = os.path.join(root_folder, f'{args["dataset"]}.pkl')
                with open(file_name, 'rb') as f:
                    dataset = pickle.load(f)
                dataset = dataset[0]
            else:
                dataset = get_dataset(name=args["dataset"], use_lcc=True)
            if args["undirected"]:
                dataset._data.edge_index = to_undirected(dataset._data.edge_index)
            dataset._data.edge_index = remove_self_loops(dataset._data.edge_index)[0]
        data = dataset._data
        data.num_classes = torch.unique(data.y).shape[0]

        return data, data.num_nodes


def get_model(args, data, n_layers):

    if args["GNN"] in ['SSobGNN', 'SobGNN']:
        model = SSobGNN(in_channels=data.num_features,
                        out_channels=data.num_classes,
                        number_layers=n_layers,
                        args=args)
    if args["GNN"] == 'GCN':
        model = GCN(in_channels=data.num_features,
                    out_channels=data.num_classes,
                    number_layers=n_layers,
                    args=args)
    if args["GNN"] == 'Cheby':
        model = Cheby(in_channels=data.num_features,
                        out_channels=data.num_classes,
                        number_layers=n_layers,
                        args=args)
    if args["GNN"] == 'kGNN':
        model = kGNN(in_channels=data.num_features,
                        out_channels=data.num_classes,
                        number_layers=n_layers,
                        args=args)
    if args["GNN"] == 'GAT':
        model = GAT(in_channels=data.num_features,
                    out_channels=data.num_classes,
                    number_layers=n_layers,
                    args=args)
    if args["GNN"] == 'Transformer':
        model = Transformer(in_channels=data.num_features,
                            out_channels=data.num_classes,
                            number_layers=n_layers,
                            args=args)
    if args["GNN"] == 'SGC':
        model = SGC(in_channels=data.num_features,
                    out_channels=data.num_classes,
                    number_layers=n_layers,
                    args=args)
    if args["GNN"] == 'ClusterGCN':
        model = ClusterGCN(in_channels=data.num_features,
                            out_channels=data.num_classes,
                            number_layers=n_layers,
                            args=args)
    if args["GNN"] == 'FiLM':
        model = FiLM(in_channels=data.num_features,
                        out_channels=data.num_classes,
                        number_layers=n_layers,
                        args=args)
    if args["GNN"] == 'SuperGAT':
        model = SuperGAT(in_channels=data.num_features,
                            out_channels=data.num_classes,
                            number_layers=n_layers,
                            args=args)
    if args["GNN"] == 'GATv2':
        model = GATv2(in_channels=data.num_features,
                        out_channels=data.num_classes,
                        number_layers=n_layers,
                        args=args)
    if args["GNN"] == 'ARMA':
        model = ARMA(in_channels=data.num_features,
                    out_channels=data.num_classes,
                    number_layers=n_layers,
                    args=args)
    if args["GNN"] == 'SIGN':
        model = SIGN(in_channels=data.num_features,
                    out_channels=data.num_classes,
                    args=args)

    return model