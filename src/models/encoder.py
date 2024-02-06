import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList, Embedding

from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\GitHub\\Molecumixer')
from utils import torchload
from tqdm import tqdm
torch.manual_seed(31415)


class CGTNN(torch.nn.Module):

    def __init__(self, feature_size, embedding_size, attention_heads, n_layers, dropout_ratio,
                 top_k_ratio, top_k_every_n, dense_neurons, edge_dim):
        super(CGTNN, self).__init__()

        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.attention_heads = attention_heads
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.top_k_ratio = top_k_ratio
        self.top_k_every_n = top_k_every_n
        self.dense_neurons = dense_neurons
        self.edge_dim = edge_dim

        self.convs = ModuleList([])
        self.transf = ModuleList([])
        self.poolings = ModuleList([])
        self.bns = ModuleList([])

        self.conv1 = TransformerConv(self.feature_size, self.embedding_size,
                                     heads=self.attention_heads, dropout=self.dropout_ratio,
                                     edge_dim=self.edge_dim,
                                     beta=True)
        self.transf1 = Linear(self.embedding_size*self.attention_heads, self.embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)


        for i in range(self.n_layers):
            self.convs.append(TransformerConv(
                self.embedding_size,
                self.embedding_size,
                heads=self.attention_heads,
                dropout=self.dropout_ratio,
                edge_dim=self.edge_dim,
                beta=True))
            self.transf.append(Linear(self.embedding_size*self.attention_heads, self.embedding_size))
            self.bns.append(BatchNorm1d(self.embedding_size))
            if i % self.top_k_every_n == 0:
                self.poolings.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.l1 = Linear(embedding_size*2, embedding_size)
    def forward(self, x, edge_attr, edge_index, batch_index):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.transf1(x))
        x = self.bn1(x)

        global_reprs = []

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.relu(self.transf[i](x))
            x = self.bns[i](x)

            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.poolings[int(i/self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                global_reprs.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        x = sum(global_reprs)

        x = F.relu(self.l1(x))
        return x


class LinearProjection(torch.nn.Module):

    def __init__(self, embedding_dim, n_o):
        super(LinearProjection, self).__init__()

        self.proj = Linear(embedding_dim, n_o)
    
    def forward(self, x):
        return self.proj(x)
if __name__ == "__main__":
    dataloader = torchload("data\\loaders\\sample_loader.moldata")

    BEST_PARAMETERS = {
        "batch_size": [128],
        "learning_rate": [0.01],
        "weight_decay": [0.0001],
        "sgd_momentum": [0.75], #0.8
        "scheduler_gamma": [0.8],
        "pos_weight": [1.3],
        "model_embedding_size": [512],
        "model_attention_heads": [3],
        "model_layers": [4],
        "model_dropout_rate": [0.2],
        "model_top_k_ratio": [0.5],
        "model_top_k_every_n": [1],
        "model_dense_neurons": [256]
    }

    model = CGTNN(feature_size=9,
                  embedding_size=BEST_PARAMETERS['model_embedding_size'][0],
                  attention_heads=BEST_PARAMETERS['model_attention_heads'][0],
                  n_layers=BEST_PARAMETERS['model_layers'][0],
                  dropout_ratio=BEST_PARAMETERS['model_dropout_rate'][0],
                  top_k_ratio=BEST_PARAMETERS['model_top_k_ratio'][0],
                  top_k_every_n=BEST_PARAMETERS['model_top_k_every_n'][0],
                  dense_neurons=BEST_PARAMETERS['model_dense_neurons'][0],
                  edge_dim=3)
    print(model)
    for i, batch in enumerate(tqdm(dataloader)):
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
