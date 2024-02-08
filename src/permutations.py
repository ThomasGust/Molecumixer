import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from tqdm import tqdm
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint, Descriptors3D, GraphDescriptors, DataStructs
from rdkit.Avalon import pyAvalonTools as avalon
from utils import load, dump, torchload, torchdump#, timeout
import torch
import math

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric
from config import X_MAP as x_map
from config import E_MAP as e_map
from stopit import threading_timeoutable as timeoutable
import numpy as np
import random
from utils import from_smiles, to_smiles
from paddings import pad_nodes, pad_edge_attr, pad_edge_index, pad_graph, pad_graph_batch

# A LOT OF THE BELOW CODE THAT DEALS WITH PERMUTATIONS WILL NEED TO GET REWORKED.
# PERMUTATIONS NEED TO BE APPLIED BEFORE PADDING. BECAUSE OF THIS, PERMUTATION FUNCTIONS WILL NEED TO
# HANDLE MATRICES OF VARYING SIZES. BY DOING IT THIS WAY, WE SHOULD ALSO BE ABLE TO ELIMINATE THE NEED
# FOR SWAPPING BACK AND FORTH BETWEEN EDGE INDEXs AND ADJACENCY MATRICES, WHICH IS NICE. #TODO

def compute_hamming_distance(v):
    """This function computes the hamming distance between a permutation vector and the base permutation
        It counts the spots that don't equal one another
    """
    c = len(v)
    base = [i for i in range(c)]
    #print(list(zip(base, v)))
    return sum([0 if v1 == v2 else 1 for v1, v2 in list(zip(base, v))])


def get_orientation_vector(n, max_distance):
    base_vector = list(range(n))
    permuted_vector = base_vector.copy()
    
    max_distance = min(max_distance, n)

    for _ in range(max_distance):
        idx1, idx2 = random.sample(range(n), 2)

        permuted_vector[idx1], permuted_vector[idx2] = permuted_vector[idx2], permuted_vector[idx1]
        current_distance = sum([1 for i, j in zip(base_vector, permuted_vector) if i != j])

        if current_distance >= max_distance:
            break
    
    return permuted_vector

def permute_n_m_matrix(matrix, new_orientation):
    # In the future, we might want this function to randomize where it places the odd numbered chunk


    n, m = matrix.shape
    num_chunks = len(new_orientation)
    
    permuted_matrix = np.zeros_like(matrix)
    chunk_height = int(math.floor(n/num_chunks))
    
    for new_pos, original_pos in enumerate(new_orientation):
        
        original_row_start = original_pos * chunk_height
        new_row_start = new_pos * chunk_height
        current_chunk_height = chunk_height if (original_pos < num_chunks - 1) else n - original_row_start
        
        permuted_matrix[new_row_start:new_row_start+current_chunk_height, :] = matrix[original_row_start:original_row_start+current_chunk_height, :]
    
    return permuted_matrix

def permute_nodes(graph, chunks, maximum_hamming_distance):
    orientation_vector = get_orientation_vector(chunks, maximum_hamming_distance)
    node_matrix = graph.x
    permuted_matrix = permute_n_m_matrix(node_matrix, orientation_vector)
    graph.x = permuted_matrix

    return graph, orientation_vector

def permute_edges(graph, chunks, maximum_hamming_distance):
    # Permutes on side of the edge index and the whole edge attributes according to a permutation vector with a fixed hamming distance
    orientation_vector = get_orientation_vector(chunks, maximum_hamming_distance)
    edge_index = torch.permute(graph.edge_index, (1,0))
    top = edge_index[:, 0]
    permuted_top = permute_n_m_matrix(top, orientation_vector)

    edge_index[:, 0] = permuted_top
    permuted_edge_index = torch.permute(edge_index, (1,0))
    graph.edge_index = permuted_edge_index

    
    edge_attributes = graph.edge_attr
    permuted_edge_attributes = permute_n_m_matrix(edge_attributes, orientation_vector)
    graph.edge_attr = permuted_edge_attributes

    return graph, orientation_vector