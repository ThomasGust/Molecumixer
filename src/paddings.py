import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\Chemix\\molecular_analysis')

import torch
import numpy as np
from utils import torchload
from data import from_smiles
from config import MAX_MOLECULE_SIZE, MAX_EDGES
import torch.nn.functional as F
#print(MAX_EDGES)
#dataloader = torchload()

# Max atoms and max edges will be used to calculate where the padding should go.
# These values are also neccessary to calculate the output dimensions of the VAE/GAN decoder.
"""
print(MAX_EDGES)

tylenol = from_smiles(smiles="CC(=O)NC1=CC=C(C=C1)O", with_hydrogen=True)
print(tylenol.size)

water = from_smiles(smiles="O", with_hydrogen=True)
print(water.size)

rand_hiv = from_smiles(smiles="CC(C)N(C(C)C)P(=O)(OP(=O)(c1ccc([N+](=O)[O-])cc1)N(C(C)C)C(C)C)c1ccc([N+](=O)[O-])cc1", with_hydrogen=True)
print(rand_hiv.size)
"""
#rand_hiv2 = from_smiles(smiles="Cc1cc(-c2ccc(N=Nc3c(S(=O)(=O)O)cc4cc(S(=O)(=O)O)cc(N)c4c3O)c(C)c2)ccc1N=Nc1c(S(=O)(=O)O)cc2cc(S(=O)(=O)O)cc(N)c2c1O", with_hydrogen=True)
#print(rand_hiv2.size)


def pad_nodes(n, v):
    ln = n.size()[0]
    padding_length = MAX_MOLECULE_SIZE - ln
    assert ln != 0
    assert padding_length >= 0

    return F.pad(input=n, pad=(0,0, 0, padding_length), mode='constant', value=v)

def pad_edge_attr(e, v):
    le = e.size()[0]
    padding_length = MAX_EDGES - le
    assert le != 0
    assert padding_length >= 0
    
    return F.pad(input=e, pad=(0, 0, 0, padding_length), mode='constant', value=v)

def pad_edge_index(e, v):
    le = e.size()[1]
    padding_length = MAX_EDGES - le
    assert le != 0
    assert padding_length >= 0
    
    return F.pad(input=e, pad=(0, padding_length, 0, 0), mode='constant', value=v)

def pad_graph(g, v):
    g.x = pad_nodes(g.x, v)
    g.edge_attr = pad_edge_attr(g.edge_attr, v)
    g.edge_index = pad_edge_index(g.edge_index, v)
    return g

def get_edge_mask_for_graph(data, graph_idx):
    # Get the source nodes of each edge
    src_nodes = data.edge_index[0]

    # Get the graph assignment for each edge using the batch tensor and source nodes
    edge_graph_assignments = data.batch[src_nodes]

    # Create a mask where the edge's graph assignment matches the desired graph_idx
    mask = edge_graph_assignments == graph_idx
    
    return mask

def pad_graph_batch(g, v, bbs):
    # IT WOULD BE NICE IF I COULD MAKE THIS FUNCTION MORE EFFICIENT
    l_x = []
    l_edge_attrs = []
    l_edge_indexes = []

    for graph_idx in torch.unique(g.batch):
        try:
            mask = g.batch == graph_idx
            edge_mask = get_edge_mask_for_graph(g, graph_idx)

            graph_x = g.x[mask]
            graph_edge_attr = g.edge_attr[edge_mask]
            graph_edge_index = g.edge_index[:, edge_mask]

            graph_x = pad_nodes(graph_x, v)
            graph_edge_attr = pad_edge_attr(graph_edge_attr, v)
            graph_edge_index = pad_edge_index(graph_edge_index, 0)

            l_x.append(graph_x)
            l_edge_attrs.append(graph_edge_attr)
            l_edge_indexes.append(graph_edge_index)
        except AssertionError:
            bbs -= 1
            #print(bbs)
    
    g.x = torch.cat(l_x, dim=0)
    g.edge_attr = torch.cat(l_edge_attrs, dim=0)
    g.edge_index = torch.cat(l_edge_indexes, dim=1)

    return g, bbs

def compute_mask(t, v):
    return t != v