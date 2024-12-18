import torch
import torch.nn as nn  # Use torch.nn for ReLU and other basic neural network components
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv
class ST_GAT(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """
    def __init__(self, in_channels, out_channels, n_nodes, heads=64, dropout=0.0):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param out_channels Number of output channels
        :param n_nodes Number of nodes in the graph
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super().__init__()
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 9 #NN: get it from config
        lstm1_hidden_size = 128   #32
        lstm2_hidden_size = 256  #128
        hidden_dim=256
        # Node-level MLP (acts as a feature transformer)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, hidden_dim), # Transform to hidden_dim space
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
        
        # single graph attentional layer with 8 attention heads
        self.gat = GATConv(in_channels=in_channels, out_channels=in_channels,
            heads=heads, dropout=0, concat=False)
        self.gcn = GCNConv(in_channels=in_channels, out_channels=in_channels, dropout=0, concat=False)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=2)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=5)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        
        # self.lstm3 = torch.nn.LSTM(input_size=lstm2_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        # for name, param in self.lstm3.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data, device):
        """
        Forward pass of the ST-GAT model
        :param data Data to make a pass on
        :param device Device to operate on
        """
        x, edge_index = data.x, data.edge_index
        # apply dropout
        if device == 'cpu':
            x = torch.FloatTensor(x)
        else:
            x = torch.cuda.FloatTensor(x)
        
        x = self.node_mlp(x)  # Node-level transformation
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        # RNN: 2 LSTM needs 3d input
        # [batchsize*n_nodes, seq_length] -> [batch_size, n_nodes, seq_length]
        batch_size = data.num_graphs
        n_node = int(data.num_nodes/batch_size)
        x = torch.reshape(x, (batch_size, n_node, data.num_features))
        # for lstm: x should be (seq_length, batch_size, n_nodes)
        # sequence length = 12, batch_size = 50, n_node = 204
        x = torch.movedim(x, 2, 0)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x) 
        # Output contains h_t for each timestep, only the last one has all input's accounted for
        x = torch.squeeze(x[-1, :, :])
        x = self.linear(x)

        # Now reshape into final output
        s = x.shape
        # [50, 204*9] -> [50, 204, 9]
        x = torch.reshape(x, (s[0], self.n_nodes, self.n_pred))
        # [batch_size, 204, 9] ->  [11400, 9]
        x = torch.reshape(x, (s[0]*self.n_nodes, self.n_pred))
        return x
