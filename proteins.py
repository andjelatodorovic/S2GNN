import argparse
import torch
import time
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import scatter,to_scipy_sparse_matrix
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import ChebConv, GCNConv, Linear
from torch_geometric.nn import GATConv,SGConv,SuperGATConv, ClusterGCNConv
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv, TransformerConv, SGConv, ClusterGCNConv, \
    FiLMConv, SuperGATConv, GATv2Conv, ARMAConv
from torch_geometric.datasets import KarateClub
from gcn.nets.layers import LinearCombinationLayer, ConcatLinearTransformationLayer,GraphConvolution
from torch.profiler import profile, record_function, ProfilerActivity
from torch_geometric.utils import subgraph
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import gc


from torch_scatter import scatter_mean

# Assuming the necessary model classes (GAT, Cheby, GCN, etc.) are imported
# from your_model_library import GAT, Cheby, GCN, SGC, SSobGNN, ClusterGCN, SuperGAT, Transformer, GATv2

def get_model(model_name, in_channels, out_channels, num_layers, args):

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if model_name == 'GAT':
        return GAT(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'Cheby':
        return Cheby(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'GCN':
        return GCN(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'SGC':
        return SGC(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'SSobGNN':
        return SSobGNN(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'ClusterGCN':
        return ClusterGCN(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'SuperGAT':
        return SuperGAT(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'Transformer':
        return Transformer(in_channels, out_channels, num_layers, args).to(device)
    elif model_name == 'GATv2':
        return GATv2(in_channels, out_channels, num_layers, args).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class Logger(object):
    def __init__(self, runs, info=None, file_name='results.txt'):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.file = open(file_name, 'a')  # Open the file in append mode

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def _write_to_file(self, message):
        self.file.write(message + '\n')
        print(message)  # Optionally print to console as well

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            self._write_to_file(f'Run {run + 1:02d}:')
            self._write_to_file(f'Highest Train: {result[:, 0].max():.2f}')
            self._write_to_file(f'Highest Valid: {result[:, 1].max():.2f}')
            self._write_to_file(f'  Final Train: {result[argmax, 0]:.2f}')
            self._write_to_file(f'   Final Test: {result[argmax, 2]:.2f}')
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

            self._write_to_file(f'All runs:')
            r = best_result[:, 0]
            self._write_to_file(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            self._write_to_file(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            self._write_to_file(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            self._write_to_file(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    def close(self):
        self.file.close()


            
class CascadeLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(CascadeLayer, self).__init__()

        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(args.alpha):
            self.convs.append(GCNConv(out_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = args.dropout

    def forward(self, x, data):
        edge_indexs, edge_attrs = data.edge_index, data.edge_attr
        hs = []


        h = self.lin(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        hs.append(h)


        for i, conv in enumerate(self.convs):
            h = conv(h, edge_indexs[i], edge_weight=edge_attrs[i])
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

            

        return hs
    
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
        edge_weight = edge_attr.mean(dim=1)  # Computing the mean across features for each edge
        x = self.conv(x, edge_index, edge_weight=edge_weight)
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

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(x)
    

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

        #self.aggregation = args["aggregation"]
        self.aggregation = 'linear'

        self.convs = torch.nn.ModuleList()
        self.concat_layers = torch.nn.ModuleList()
        self.convs.append(CascadeLayer(in_channels, args.hidden_channels, args))
        for _ in range(number_layers - 2):
            self.convs.append(CascadeLayer(args.hidden_channels, args.hidden_channels, args))
        self.convs.append(CascadeLayer(args.hidden_channels, out_channels, args))
        #self.convs.append(torch.nn.Linear(2,out_channels))
        
        #self.convs.append(torch.nn.Linear(args.hidden_channels,out_channels))

        if args.aggregation == 'linear':
            self.linear_combination_layers = torch.nn.ModuleList()
            for _ in range(number_layers):
                self.linear_combination_layers.append(LinearCombinationLayer(args.alpha))
        if args.aggregation == 'concat':
            self.concat_layers = torch.nn.ModuleList()
            for _ in range(number_layers-1):
                self.concat_layers.append(ConcatLinearTransformationLayer(args.alpha, args.hidden_channels, args.hidden_channels))
            self.concat_layers.append(ConcatLinearTransformationLayer(args.alpha, out_channels, out_channels))


        
    def forward(self, data):
        x = data.x
        for i, conv_layer in enumerate(self.convs):
            hs = conv_layer(data.x, data)
            if self.aggregation == 'linear':
                x = self.linear_combination_layers[i](hs)
            if self.aggregation == 'concat':
                x = self.concat_layers[i](hs)
        return x.log_softmax(dim=-1)
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GAT, self).__init__()

        # Assuming args contains the heads, hidden_channels, and dropout values
        self.heads = 1  # Reduce the number of heads to reduce memory usage
        self.hidden_units = max(8, args.hidden_channels // 2)  # Reduce hidden units
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
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        
        # Final layer

        x = self.convs[-1](x, edge_index)

        return torch.sigmoid(x)
    
class Transformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(Transformer, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, args.hidden_channels, heads=1, concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(TransformerConv(args.hidden_channels, args.hidden_channels, heads=1, concat=False))
        self.convs.append(TransformerConv(args.hidden_channels, out_channels, heads=1, concat=False))

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Use the last convolutional layer
        x = self.convs[-1](x, edge_index)

        # Apply sigmoid activation for multi-label classification
        return torch.sigmoid(x)

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GATv2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, args.hidden_channels, heads=1, concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(GATv2Conv(args.hidden_channels, args.hidden_channels, heads=1, concat=False))
        self.convs.append(GATv2Conv(args.hidden_channels, out_channels, heads=1, concat=False))

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Using sigmoid for multi-label classification
        return torch.sigmoid(x)
    

def train(model, data, y_true, train_idx, optimizer, scaler):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    with autocast():
        out = model(data)[train_idx]
        loss = criterion(out, y_true[train_idx].float())

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    gc.collect()

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

    
def create_subgraph(data, subset_percentage):
    # Calculate the number of nodes in the subset
    #num_nodes_subset = int(subset_percentage * data.num_nodes)
    num_nodes_subset = 30000
    # Select a subset of nodes based on the calculated number
    subset_nodes = torch.arange(num_nodes_subset)

    # Extract the subgraph
    subset_edge_index, subset_edge_attr = subgraph(subset_nodes, data.edge_index, edge_attr=data.edge_attr, num_nodes=data.num_nodes)
    print(subset_edge_index.shape)
    # Create a new data object for the subgraph
    subset_data = torch_geometric.data.Data(
        x=data.x[subset_nodes],
        edge_index=subset_edge_index,
        edge_attr=subset_edge_attr,
        y=data.y[subset_nodes]
    )

    return subset_data

def random_split(data, train_percent, val_percent, test_percent):
    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)
    
    train_size = int(train_percent * num_nodes)
    val_size = int(val_percent * num_nodes)
    
    train_idx = torch.tensor(indices[:train_size], dtype=torch.long)
    val_idx = torch.tensor(indices[train_size:train_size + val_size], dtype=torch.long)
    test_idx = torch.tensor(indices[train_size + val_size:], dtype=torch.long)
    
    return {'train': train_idx, 'valid': val_idx, 'test': test_idx}


def main():
    

    parser = argparse.ArgumentParser(description='OGBN-Proteins (Cheby)')
    parser.add_argument('--model', type=str, default='SGC',
                            help='Name of graph neural network: SSobGNN, SobGNN, GCN, Cheby, kGNN,'
                                'GAT, Transformer, SGC, ClusterGCN, FiLM, SuperGAT, GATv2, ARMA, SIGN')
    parser.add_argument('--aggregation', type=str, default='linear')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=11)
    parser.add_argument('--K_Cheby', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=6)
    args = parser.parse_args()
    print(args)

    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins')
    
    data = dataset[0]
    # Example usage
    
    x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean').to(device)

  
    data.x = x
    data = create_subgraph(data, subset_percentage=0.3)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.y = data.y.to(device)

    
    if args.model == 'SSobGNN':
        adj = to_scipy_sparse_matrix(data.edge_index)
        new_edges = []
        for i in range(adj.shape[0]):  # Changed from 1 to 0 to ensure correct indexing
            new_edges_tensor = torch.tensor([[i, i]], dtype=torch.long).t()
            new_edges_tensor = new_edges_tensor.to(device)  # Move tensor to the same device as data
            new_edges.append(new_edges_tensor)
        
        edge_index_temp = torch.cat([data.edge_index] + new_edges, dim=1)
        edge_attributes = []
        edge_index = []
        alpha = 6
        epsilon = 1.5 
        for rho in range(1, alpha + 1):
            edge_index.append(edge_index_temp)
            sparse_sobolev_term = torch.pow(torch.full_like(edge_index_temp[0], epsilon, dtype=torch.float), rho)
            sparse_sobolev_term = sparse_sobolev_term.to(device)  # Ensure correct device and dtype
            edge_attributes.append(sparse_sobolev_term)
    
        data.edge_attr = edge_attributes
        data.edge_index = edge_index

    
    # Use random_node_split for splitting the dataset
    split_idx = random_split(data, train_percent=0.6, val_percent=0.2, test_percent=0.2)

    # Now you can use train_idx, val_idx, and test_idx in your training and evaluation loops
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    torch.cuda.empty_cache()
    
    model = get_model(args.model, x.size(-1), 112, args.num_layers, args).to(device)


    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)
    scaler = GradScaler()

    total_time_excluding_first_run = 0
    num_runs_excluding_first = args.runs - 1 if args.runs > 1 else 0

    for run in range(args.runs):
        start_time = time.time()  # Start time of the run

        model = get_model(args.model, x.size(-1), 112, args.num_layers, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, data.y, train_idx, optimizer, scaler)

            if epoch % args.eval_steps == 0:
                result = test(model, data, data.y, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')


        run_time = time.time() - start_time  # Time for this run
        print(f'Time for run {run + 1:02d}: {run_time:.2f} seconds')

        if run > 0:  # Exclude the first run
            total_time_excluding_first_run += run_time

        logger.print_statistics(run)

    # Average time calculation
    average_time_per_run = total_time_excluding_first_run / num_runs_excluding_first if num_runs_excluding_first > 0 else 0
    print(f'Average time per run (excluding first run): {average_time_per_run:.2f} seconds')

    logger.print_statistics()
    logger.close()

if __name__ == "__main__":
    main()
    