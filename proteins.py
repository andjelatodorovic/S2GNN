import argparse
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.nn import GATConv,SGConv,SuperGATConv, ClusterGCNConv
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv, TransformerConv, SGConv, ClusterGCNConv, \
    FiLMConv, SuperGATConv, GATv2Conv, ARMAConv

from gcn.nets.layers import CascadeLayer, LinearCombinationLayer, ConcatLinearTransformationLayer,GraphConvolution

from torch_scatter import scatter_mean

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

class Cheby(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(Cheby, self).__init__()

        self.K = args.K_Cheby

        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(in_channels, args.hidden_channels, K=self.K))
        for _ in range(number_layers - 2):
            self.convs.append(ChebConv(args.hidden_channels, args.hidden_channels, K=self.K))
        self.convs.append(ChebConv(args.hidden_channels, out_channels, K=self.K))

        self.dropout = args.dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.log_softmax(x, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GAT, self).__init__()

        self.heads = 1
        self.hidden_units = args.hidden_channels
        self.dropout = args.dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, self.hidden_units, heads=self.heads, concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(GATConv(self.hidden_units, self.hidden_units, heads=self.heads, concat=False))
        self.convs.append(GATConv(self.hidden_units, out_channels, heads=self.heads, concat=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            print(f"Before conv {i + 1}: x.shape = {x.shape}, edge_index.shape = {edge_index.shape}, edge_attr.shape = {edge_attr.shape}")
            x = conv(x, edge_index)
            print(f"After conv {i + 1}: x.shape = {x.shape}")
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, args.hidden_channels, cached=True))
        for _ in range(number_layers - 2):
            self.convs.append(GCNConv(args.hidden_channels, args.hidden_channels, cached=True))
        self.convs.append(GCNConv(args.hidden_channels, out_channels, cached=True))

        self.dropout = args.dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
    
class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(SGC, self).__init__()

        self.conv = SGConv(in_channels, out_channels, K=number_layers, cached=True)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=-1)
    
class SuperGAT(torch.nn.Module):
    r"""Parametrized SuperGAT model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(SuperGAT, self).__init__()
        self.headsAttention = 1
        self.convs = torch.nn.ModuleList()
        self.convs.append(SuperGATConv(in_channels, args.hidden_channels, heads=self.headsAttention, concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(SuperGATConv(args.hidden_channels, args.hidden_channels, heads=self.headsAttention, concat=False))
        self.convs.append(SuperGATConv(args.hidden_channels, out_channels, heads=self.headsAttention, concat=False))

        self.dropout = args.dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

class ClusterGCN(torch.nn.Module):
    r"""Parametrized ClusterGCNConv model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(ClusterGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ClusterGCNConv(in_channels, args.hidden_channels))
        for _ in range(number_layers - 2):
            self.convs.append(ClusterGCNConv(args.hidden_channels, args.hidden_channels))
        self.convs.append(ClusterGCNConv(args.hidden_channels, out_channels))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)
    

class SSobGNN(torch.nn.Module):
    r"""Parametrized S-SobGNN model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(SSobGNN, self).__init__()

        self.aggregation = args["aggregation"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(CascadeLayer(in_channels, args["hidden_units"], args))
        for _ in range(number_layers - 2):
            self.convs.append(CascadeLayer(args["hidden_units"], args["hidden_units"], args))
        self.convs.append(CascadeLayer(args["hidden_units"], out_channels, args))
        #self.convs.append(torch.nn.Linear(2,out_channels))
        
        #self.convs.append(torch.nn.Linear(args["hidden_units"],out_channels))

        if args["aggregation"] == 'linear':
            self.linear_combination_layers = torch.nn.ModuleList()
            for _ in range(number_layers):
                self.linear_combination_layers.append(LinearCombinationLayer(args["alpha"]))
        if args["aggregation"] == 'concat':
            self.concat_layers = torch.nn.ModuleList()
            for _ in range(number_layers-1):
                self.concat_layers.append(ConcatLinearTransformationLayer(args["alpha"], args["hidden_units"], args["hidden_units"]))
            self.concat_layers.append(ConcatLinearTransformationLayer(args["alpha"], out_channels, out_channels))


        
    def forward(self, data):
        x = data.x
        for i, conv_layer in enumerate(self.convs):
            hs = conv_layer(data.x, data)
            if self.aggregation == 'linear':
                x = self.linear_combination_layers[i](hs)
            if self.aggregation == 'concat':
                x = self.concat_layers[i](hs)
        return x.log_softmax(dim=-1)

def train(model, data, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data)[train_idx]
    loss = criterion(out, y_true[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, y_true, split_idx, evaluator):
    model.eval()

    y_pred = model(data)

    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

    '''
    adj = to_scipy_sparse_matrix(data.edge_index)
    new_edges = []
    for i in range(1,adj.shape[0]+1):
         new_edges.append(torch.tensor([[i, i]], dtype=torch.long).t())
    
    edge_index_temp = torch.cat([data.edge_index] + new_edges, dim=1)
    data.edge_attrs = []
    edge_attributes = []
    edge_index = []
    alpha = 6
    epsilon = 1.5 
    for rho in range(1, alpha+1):
         edge_index.append(edge_index_temp)
         sparse_sobolev_term = torch.pow(torch.full_like(edge_index_temp[0], epsilon), rho)
         edge_attributes.append(sparse_sobolev_term.to(float))
    
    '''

def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (Cheby)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--K_Cheby', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean').to(device)

    '''
    adj = to_scipy_sparse_matrix(data.edge_index)
    new_edges = []
    for i in range(1,adj.shape[0]+1):
         new_edges.append(torch.tensor([[i, i]], dtype=torch.long).t())
    
    edge_index_temp = torch.cat([data.edge_index] + new_edges, dim=1)
    data.edge_attrs = []
    edge_attributes = []
    edge_index = []
    alpha = 6
    epsilon = 1.5 
    for rho in range(1, alpha+1):
         edge_index.append(edge_index_temp)
         sparse_sobolev_term = torch.pow(torch.full_like(edge_index_temp[0], epsilon), rho)
         edge_attributes.append(sparse_sobolev_term.to(float))
    
    '''

    data.x = x
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.y = data.y.to(device)
    
    train_idx = split_idx['train'].to(device)
    torch.cuda.empty_cache()
    model = Cheby(x.size(-1), 112, args.num_layers, args).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        #model.reset_parameters()
        print(torch.cuda.memory_summary(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, data.y, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, data, data.y, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc= result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()