import argparse


def parse_args(train_mode=True):

    # Arguments to run the experiment
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--GraphDifussion", action='store_true', default=False,
                    help='Activate Graph Difussion preprocessing.')
    parser.add_argument('--GNN', type=str, default='SSobGNN',
                        help='Name of graph neural network: SSobGNN, SobGNN, GCN, Cheby, kGNN,'
                            'GAT, Transformer, SGC, ClusterGCN, FiLM, SuperGAT, GATv2, ARMA, SIGN')
    parser.add_argument('--dataset', type=str, default='cancer_b',#Citeseer',
                help='Name of dataset, options: {cancer_b, cancer, 20news, activity,'
                    'isolet, Cornell, Texas, Wisconsin, chameleon,'
                    'Actor, squirrel, Cora, Citeseer, Pubmed, COCO-S, PascalVOC-SP}')
    parser.add_argument('--graph', type=str, default='knn',
                        help='knn or learned graph')
    parser.add_argument('--gpu_number', type=int, default=0,
                    help='GPU index.')
    parser.add_argument("--alphaGDC", type=float, default=0.05,
                            help='alpha value for graph difussion method.')
    parser.add_argument("--k", type=int, default=128,
                        help='k value for graph difussion method.')
    parser.add_argument('--undirected', action='store_true', default=True,
                        help='set to not symmetrize adjacency')
    parser.add_argument('--verbose', action='store_true', default=False,
                            help='Show results.')
    parser.add_argument('--epochs', type=int, default=5,#00,
                        help='Number of epochs to train.')

    # train
    if train_mode:
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden_units', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument("--n_layers", type=int, default=3,
                            help='List of number of layers.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--aggregation', type=str, default='linear',
                            help='Aggregation type for S2-GNN: linear, concat, attention')
        parser.add_argument("--K_Cheby", type=int, default=1,
                            help='K value for ChebConv.')
        parser.add_argument("--K_SIGN", type=int, default=2,
                            help='K value for SIGN.')
        parser.add_argument('--alpha', type=int, default=6,
                            help='Parameter alpha for S-SobGNN.')
        parser.add_argument('--headsAttention', type=int, default=1,
                            help='Parameter heads for GAT, Transformer, SuperGAT, and GATv2.')
        parser.add_argument('--epsilon', type=float, default=1.5,
                            help='Parameter epsilon for S-SobGNN or SobGNN.')
        parser.add_argument("--hyperparameterTunning_mode", action='store_true', default=False,
                            help='Activate hyperparameter tunning mode.')
    else:
        parser.add_argument("--hyperparameterTunning_mode", action='store_true', default=True,
                            help='Activate hyperparameter tunning mode.')
        parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations random search.')

    args = parser.parse_args()
    print("args = ", args)

    return vars(args) # returns a dictionary
