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
    print(list(zip(base, v)))
    return sum([0 if v1 == v2 else 1 for v1, v2 in list(zip(base, v))])


def permute_hamming_vector(n, max_distance):
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

"""
def permute_square_matrix(matrix, new_orientation, chunk_size):
    
    #Permute chunks of an (nXn) matrix to match a new orientation (n must be divisible by chunk_size)
    n = matrix.shape[0] 
    num_chunks_per_side = n // chunk_size
    
    permuted_matrix = np.zeros_like(matrix)
    
    for new_pos, original_pos in enumerate(new_orientation):
        
        original_row = (original_pos // num_chunks_per_side) * chunk_size
        original_col = (original_pos % num_chunks_per_side) * chunk_size
        
        new_row = (new_pos // num_chunks_per_side) * chunk_size
        new_col = (new_pos % num_chunks_per_side) * chunk_size
        
        permuted_matrix[new_row:new_row+chunk_size, new_col:new_col+chunk_size] = \
            matrix[original_row:original_row+chunk_size, original_col:original_col+chunk_size]
    
    return permuted_matrix

def permute_matrix_chunks_3d(matrix, new_orientation, chunk_size):
    #Permute chunks of a 3D matrix (n x n x m) to match a new orientation, keeping the third dimension intact.
    n, _, m = matrix.shape  # Assuming the matrix has shape (n, n, m)
    num_chunks_per_side = n // chunk_size
    
    permuted_matrix = np.zeros_like(matrix)
    
    for new_pos, original_pos in enumerate(new_orientation):
        original_row = (original_pos // num_chunks_per_side) * chunk_size
        original_col = (original_pos % num_chunks_per_side) * chunk_size
        new_row = (new_pos // num_chunks_per_side) * chunk_size
        new_col = (new_pos % num_chunks_per_side) * chunk_size
        
        permuted_matrix[new_row:new_row+chunk_size, new_col:new_col+chunk_size, :] = \
            matrix[original_row:original_row+chunk_size, original_col:original_col+chunk_size, :]
    
    return permuted_matrix

def permute_horizontal_chunks(matrix, new_orientation, chunk_height):
    #Permute horizontal chunks of a 2D matrix (n x m) while keeping the m dimension intact.
    n, m = matrix.shape
    num_chunks = n // chunk_height
    
    permuted_matrix = np.zeros_like(matrix)
    
    for new_pos, original_pos in enumerate(new_orientation):
        original_row_start = original_pos * chunk_height
        new_row_start = new_pos * chunk_height
        
        permuted_matrix[new_row_start:new_row_start+chunk_height, :] = matrix[original_row_start:original_row_start+chunk_height, :]
    
    return permuted_matrix

"""
# MAXIMUM HAMMING DISTANCE IN ALL OF THESE FUNCTIONS WILL BE DECIDED AT TRAINING TIME

def permute_nodes(graph, chunks, maximum_hamming_distance):
    permuted_vector = permute_hamming_vector(chunks, maximum_hamming_distance)
    node_matrix = graph.x
    permuted_matrix = permute_node_matrix(node_matrix, permuted_vector)
    graph.x = permuted_matrix

    return graph

def permute_edges(graph, chunks, maximum_hamming_distance):
    permuted, label = None, None
    return permuted, label

def permute_graph(graph, chunks, maximum_hamming_distance):
    permuted, label = None, None
    return permuted, label

def calculate_descriptors(mol):
    o = list(Descriptors.CalcMolDescriptors(mol).values())
    return o

def calculate_graph_descriptors(mol):
    balabanJ = GraphDescriptors.BalabanJ(mol)
    bertzCT = GraphDescriptors.BertzCT(mol)
    
    chi0 = GraphDescriptors.Chi0(mol)
    chi1 = GraphDescriptors.Chi1(mol)
    chi0v = GraphDescriptors.Chi0v(mol)
    chi1v = GraphDescriptors.Chi1v(mol)
    chi2v = GraphDescriptors.Chi2v(mol)
    chi3v = GraphDescriptors.Chi3v(mol)
    chi4v = GraphDescriptors.Chi4v(mol)
    chi0n = GraphDescriptors.Chi0n(mol)
    chi1n = GraphDescriptors.Chi1n(mol)
    chi2n = GraphDescriptors.Chi2n(mol)
    chi3n = GraphDescriptors.Chi3n(mol)
    chi4n = GraphDescriptors.Chi4n(mol)

    hka = GraphDescriptors.HallKierAlpha(mol)
    ipc = GraphDescriptors.Ipc(mol)

    k1 = GraphDescriptors.Kappa1(mol)
    k2 = GraphDescriptors.Kappa2(mol)
    k3 = GraphDescriptors.Kappa3(mol)

    o = [balabanJ,bertzCT,chi0,chi1,chi0v,chi1v,chi2v,chi3v,chi4v,chi0n,chi1n,chi2n,chi3n,chi4n,hka,ipc,k1,k2,k3]
    return o
def calculate_3d_descriptors(mol):
    AllChem.EmbedMolecule(mol)

    asphericity = Descriptors3D.Asphericity(mol)
    eccentricity = Descriptors3D.Eccentricity(mol)
    isf = Descriptors3D.InertialShapeFactor(mol)

    npr1 = Descriptors3D.NPR1(mol)
    npr2 = Descriptors3D.NPR2(mol)

    pmi1 = Descriptors3D.PMI1(mol)
    pmi2 = Descriptors3D.PMI2(mol)
    pmi3 = Descriptors3D.PMI3(mol)


    rg = Descriptors3D.RadiusOfGyration(mol)

    o = [asphericity, eccentricity, isf, npr1, npr2, pmi1, pmi2, pmi3, rg]
    return o

def calculate_fingerprints(mol):
    mfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    mfp3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3)

    maccs = MACCSkeys.GenMACCSKeys(mol)
    rdkfp = RDKFingerprint(mol)

    avfp = avalon.GetAvalonFP(mol)

    return [mfp2, mfp3, maccs, rdkfp, avfp]

def row_normalize_array(a):
    a = np.clip(a, a_min=-1000, a_max=1000)
    avgs = np.mean(a, axis=1)[:,np.newaxis]
    a /= avgs
    return a

@timeoutable()
def calculate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    descriptors = np.array(calculate_descriptors(mol))
    descriptors3d = np.array(calculate_3d_descriptors(mol))
    graph_descriptors = np.array(calculate_graph_descriptors(mol))
    fingerprints = [np.frombuffer(fingerprint.ToBitString().encode(), 'u1') - ord('0') for fingerprint in calculate_fingerprints(mol)]

    return descriptors, descriptors3d, graph_descriptors, fingerprints
    #return descriptors3d

def compute_sample():
    """
    As of right now, this function is super inefficient and really needs to be optimized. Right now,
    I am just trying to get feature normalization to work so I can use the descriptor data properly,
    but this function is definetly on the #TODO
    
    """
    data = load("data\\graphs\\sample_graphs.mol")
    d = []
    cntr = 0

    _descriptors = []
    _descriptors3d = []
    _graph_descriptors = []
    _fingerprints = []
    _smiles = []
    _graphs = []
    _cntr = []

    for graph, smiles in tqdm(data[:5000]):
        try:
            descriptors, descriptors3d, graph_descriptors, fingerprints = calculate(smiles, timeout=5)
            _descriptors.append(descriptors)
            _descriptors3d.append(descriptors3d)
            _graph_descriptors.append(graph_descriptors)
            _fingerprints.append(fingerprints)
            _smiles.append(smiles)
            _graphs.append(graph)
            _cntr.append(cntr)

        except Exception as e:
            print(e)
        cntr += 1
    
    print(np.array(_descriptors).shape)
    _descriptors = row_normalize_array(np.array(_descriptors))
    _descriptors3d = row_normalize_array(np.array(_descriptors3d))
    _graph_descriptors = row_normalize_array(np.array(_graph_descriptors))

    d = (_descriptors, _descriptors3d, _graph_descriptors, _fingerprints, _smiles, _graphs, _cntr)
    dump("data\\processed_graphs\\sample_graphs_5k.pmol",d)

def fetch_dataloader(pmol_path, bs=32, shuffle=True, sp=None, fpdtype=np.uint8):
    """
    This function is also in desperate need of optimization, but getting feature normalization
    working is more important right now.
    """
    data = load(pmol_path)
    _descriptors, _descriptors3d, _graph_descriptors, _fingerprints, _smiles, _graphs, _cntr = data

    new_graphs = []
    for i, graph in enumerate(tqdm(_graphs, "CREATING DATALOADER |")):

        fingerprints = _fingerprints[i]
        descriptors = _descriptors[i]
        descriptors3d = _descriptors3d[i]
        graph_descriptors = _graph_descriptors[i]

        mfp2, mfp3, maccs, rdkfp, avfp = tuple(fingerprints)

        descriptors = np.array(descriptors, dtype=np.float64)
        descriptors3d = np.array(descriptors3d, dtype=np.float64)
        graph_descriptors = np.array(graph_descriptors, dtype=np.float64)
        
        mfp2 = np.array(mfp2, dtype=fpdtype)
        mfp3 = np.array(mfp3, dtype=fpdtype)
        maccs = np.array(maccs, dtype=fpdtype)
        rdkfp = np.array(rdkfp, dtype=fpdtype)
        avfp = np.array(avfp, dtype=fpdtype)

        graph.descriptors = descriptors
        graph.descriptors3d = descriptors3d
        graph.graph_descriptors = graph_descriptors

        graph.mfp2 = mfp2
        graph.mfp3 = mfp3
        graph.maccs = maccs
        graph.rdkfp = rdkfp
        graph.avfp = avfp

        new_graphs.append(graph)
    
    dataloader = DataLoader(new_graphs, batch_size=bs, shuffle=shuffle)

    if sp is not None:
        torchdump(sp, dataloader)
    
    return dataloader

if __name__ == "__main__":
    #compute_sample()
    
    sp = "data\\loaders\\sample_loader.moldata"
    print("FETCHING DATALOADER")
    sg_path = "data\\processed_graphs\\sample_graphs.pmol"
    data_loader = fetch_dataloader(sg_path, sp=sp)
    
    
    #torchload(sp)
    batch = next(iter(data_loader))
    print(batch.x.shape)
    print(batch.edge_index)
    """
    print(np.array(batch[0].x))
    print()
    permuted = permute_nodes(batch[0], 4, 2)
    print(permuted.x)
    """
    print("TEST")
    print(batch.x.shape)
    print(batch.edge_index)

