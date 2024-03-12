import pickle as pkl
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.utils import to_dense_adj
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import ELEMENT_BASE as element_base
from config import MAX_MOLECULE_SIZE
from config import E_MAP
from typing import Any
from scipy.sparse import csr_matrix
import numpy as np
from config import X_MAP
import torch_geometric
from torch_geometric.data import Data

ATOMIC_NUMBERS =  list(range(0, 119))
SUPPORTED_ATOMS = [element_base[i][0] for i in ATOMIC_NUMBERS]
SUPPORTED_EDGES = E_MAP['bond_type']
DISABLE_RDKIT_WARNINGS = False

def edge_index_to_sparse_adjacency(edge_index, num_nodes):
    source_nodes = edge_index[0, :]
    target_nodes = edge_index[1, :]

    data = np.ones(edge_index.shape[1], dtype=int)
    adjacency_matrix = csr_matrix((data, (source_nodes, target_nodes)), shape=(num_nodes, num_nodes))

    return adjacency_matrix

def dump(path, obj):
    with open(path, "wb") as f:
        pkl.dump(obj, f)

def load(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data

def torchdump(path, obj):
    torch.save(obj, path)

def torchload(path):
    with open(path, "rb") as f:
        data = torch.load(f)
    return data

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
        x.append(X_MAP['atomic_num'].index(atom.GetAtomicNum()))
        x.append(X_MAP['chirality'].index(str(atom.GetChiralTag())))
        x.append(X_MAP['degree'].index(atom.GetTotalDegree()))
        x.append(X_MAP['formal_charge'].index(atom.GetFormalCharge()))
        x.append(X_MAP['num_hs'].index(atom.GetTotalNumHs()))
        x.append(X_MAP['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(X_MAP['hybridization'].index(str(atom.GetHybridization())))
        x.append(X_MAP['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(X_MAP['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(E_MAP['bond_type'].index(str(bond.GetBondType())))
        e.append(E_MAP['stereo'].index(str(bond.GetStereo())))
        e.append(E_MAP['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def avg(l):
    return sum(l)/len(l)

def filter_inf(t, nan=0.0, pinf=0.0, ninf=0.0):
    return torch.nan_to_num(t, nan=nan, posinf=pinf, neginf=ninf)

def is_rational(mol_g):
    try:
        _ = to_smiles(mol_g)
        return True
    except Exception:
        return False

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
        atom.SetFormalCharge(X_MAP['formal_charge'][data.x[i, 3].item()])
        atom.SetNumExplicitHs(X_MAP['num_hs'][data.x[i, 4].item()])
        atom.SetNumRadicalElectrons(
            X_MAP['num_radical_electrons'][data.x[i, 5].item()])
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
