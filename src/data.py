import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from tqdm import tqdm
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint, Descriptors3D, GraphDescriptors, DataStructs
from rdkit.Avalon import pyAvalonTools as avalon
from utils import load, dump, torchload, torchdump, timeout
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric
from config import X_MAP as x_map
from config import E_MAP as e_map
from stopit import threading_timeoutable as timeoutable
import numpy as np
import random

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

def permute_nodes(graph, chunks, maximum_hamming_distance):
    orientation = list(range(chunks**2))
    permuted_vector = permute_hamming_vector(len(orientation), maximum_hamming_distance)

def permute_edges(graph, chunks, maximum_hamming_distance):
    permuted, label = None, None
    return permuted, label

def permute_graph(graph, chunks, maximum_hamming_distance):
    permuted, label = None, None
    return permuted, label

def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)



def to_smiles(data: 'torch_geometric.data.Data',
              kekulize: bool = False) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    for i in range(data.num_nodes):
        atom = Chem.Atom(data.x[i, 0].item())
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[data.x[i, 1].item()])
        atom.SetFormalCharge(x_map['formal_charge'][data.x[i, 3].item()])
        atom.SetNumExplicitHs(x_map['num_hs'][data.x[i, 4].item()])
        atom.SetNumRadicalElectrons(
            x_map['num_radical_electrons'][data.x[i, 5].item()])
        atom.SetHybridization(
            Chem.rdchem.HybridizationType.values[data.x[i, 6].item()])
        atom.SetIsAromatic(data.x[i, 7].item())
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[data.edge_attr[i, 0].item()]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[data.edge_attr[i, 1].item()]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2].item())
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return Chem.MolToSmiles(mol, isomericSmiles=True)

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
    
    #sp = "data\\loaders\\sample_loader.moldata"
    
    #sg_path = "data\\processed_graphs\\sample_graphs.pmol"
    #data_loader = fetch_dataloader(sg_path, sp=sp)
    
    
    #torchload(sp)
    hamming_distance = compute_hamming_distance([0, 1, 2, 3, 4, 5])
    print(hamming_distance)