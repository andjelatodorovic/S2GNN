__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import torch
import torch.nn.functional as functional
from torch.nn import Linear

from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv, TransformerConv, SGConv, ClusterGCNConv, \
    FiLMConv, SuperGATConv, GATv2Conv, ARMAConv

from gcn.nets.layers import CascadeLayer, LinearCombinationLayer, ConcatLinearTransformationLayer


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
            hs = conv_layer(x, data)
            if self.aggregation == 'linear':
                x = self.linear_combination_layers[i](hs)
            if self.aggregation == 'concat':
                x = self.concat_layers[i](hs)
        return x.log_softmax(dim=-1)

class SIGN(torch.nn.Module):
    r"""Parametrized SIGN model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, args):
        super(SIGN, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(args["K_SIGN"]):
            self.convs.append(Linear(in_channels, args["hidden_units"]))
        self.lin = Linear(args["K_SIGN"] * args["hidden_units"], out_channels)
        '''
        for _ in range(args["K_SIGN"] + 1):
            self.convs.append(Linear(in_channels, args["hidden_units"]))
        self.lin = Linear((args["K_SIGN"] + 1) * args["hidden_units"], out_channels)
        '''

        self.dropout = args["dropout"]

    def forward(self, data):
        xs = data.xs
        hs = []
        for x, lin in zip(xs, self.convs):
            h = lin(x)
            h = functional.relu(h)
            h = functional.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        h = self.lin(h)
        return h.log_softmax(dim=-1)

class GCN(torch.nn.Module):
    r"""Parametrized GCN model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, args["hidden_units"], cached=True))
        for _ in range(number_layers - 2):
            self.convs.append(GCNConv(args["hidden_units"], args["hidden_units"], cached=True))
        self.convs.append(GCNConv(args["hidden_units"], out_channels, cached=True))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class Cheby(torch.nn.Module):
    r"""Parametrized ChebyConv model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(Cheby, self).__init__()

        self.K = args["K_Cheby"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(in_channels, args["hidden_units"], K=self.K))
        for _ in range(number_layers - 2):
            self.convs.append(ChebConv(args["hidden_units"], args["hidden_units"], K=self.K))
        self.convs.append(ChebConv(args["hidden_units"], out_channels, K=self.K))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class kGNN(torch.nn.Module):
    r"""Parametrized k-GNN model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(kGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, args["hidden_units"]))
        for _ in range(number_layers - 2):
            self.convs.append(GraphConv(args["hidden_units"], args["hidden_units"]))
        self.convs.append(GraphConv(args["hidden_units"], out_channels))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    r"""Parametrized GAT model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, args["hidden_units"], heads=args["headsAttention"], concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(GATConv(args["hidden_units"], args["hidden_units"], heads=args["headsAttention"], concat=False))
        self.convs.append(GATConv(args["hidden_units"], out_channels, heads=args["headsAttention"], concat=False))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class Transformer(torch.nn.Module):
    r"""Parametrized Transformer model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(Transformer, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, args["hidden_units"], heads=args["headsAttention"], concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(TransformerConv(args["hidden_units"], args["hidden_units"], heads=args["headsAttention"], concat=False))
        self.convs.append(TransformerConv(args["hidden_units"], out_channels, heads=args["headsAttention"], concat=False))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class SGC(torch.nn.Module):
    r"""Parametrized SGC model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(SGC, self).__init__()

        self.conv = SGConv(in_channels, out_channels, K=number_layers, cached=True)
        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_attr)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)
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
        self.convs.append(ClusterGCNConv(in_channels, args["hidden_units"]))
        for _ in range(number_layers - 2):
            self.convs.append(ClusterGCNConv(args["hidden_units"], args["hidden_units"]))
        self.convs.append(ClusterGCNConv(args["hidden_units"], out_channels))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class FiLM(torch.nn.Module):
    r"""Parametrized FiLM model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(FiLM, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(in_channels, args["hidden_units"]))
        for _ in range(number_layers - 2):
            self.convs.append(FiLMConv(args["hidden_units"], args["hidden_units"]))
        self.convs.append(FiLMConv(args["hidden_units"], out_channels))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type=edge_attr)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


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

        self.convs = torch.nn.ModuleList()
        self.convs.append(SuperGATConv(in_channels, args["hidden_units"], heads=args["headsAttention"], concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(SuperGATConv(args["hidden_units"], args["hidden_units"], heads=args["headsAttention"], concat=False))
        self.convs.append(SuperGATConv(args["hidden_units"], out_channels, heads=args["headsAttention"], concat=False))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class GATv2(torch.nn.Module):
    r"""Parametrized GATv2 model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GATv2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, args["hidden_units"], heads=args["headsAttention"], concat=False))
        for _ in range(number_layers - 2):
            self.convs.append(GATv2Conv(args["hidden_units"], args["hidden_units"], heads=args["headsAttention"], concat=False))
        self.convs.append(GATv2Conv(args["hidden_units"], out_channels, heads=args["headsAttention"], concat=False))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)


class ARMA(torch.nn.Module):
    r"""Parametrized ARMA model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(ARMA, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ARMAConv(in_channels, args["hidden_units"]))
        for _ in range(number_layers - 2):
            self.convs.append(ARMAConv(args["hidden_units"], args["hidden_units"]))
        self.convs.append(ARMAConv(args["hidden_units"], out_channels))

        self.dropout = args["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)