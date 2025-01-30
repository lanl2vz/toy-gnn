import torch
import torch.nn as nn
import torch_geometric

class TOYGNN(nn.Module):
    def __init__(self, num_node_features, num_layers, hidden_dim, num_classes):
        super(TOYGNN, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(torch_geometric.nn.GCNConv(num_node_features, hidden_dim))
            elif i == num_layers - 1:
                self.convs.append(torch_geometric.nn.GCNConv(hidden_dim, num_classes))
            else:
                self.convs.append(torch_geometric.nn.GCNConv(hidden_dim, hidden_dim))

        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.relu(x)

        return x
