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

from orientations import permute_edges, permute_nodes, get_orientation_vector

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

    batch = next(iter(data_loader))
    mol1 = batch[0]

    permuted_nodes, _= permute_nodes(mol1, 5, 3)
    permuted_edges = permute_edges(mol1, 5, 3)
    
    print(mol1.edge_index.shape, torch.permute(torch.tensor(permuted_edges['edge_index']), (1, 0)).shape)