from typing import Any
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from config import node_matrix_map, edge_matrix_map
import torch

import torch_geometric
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

class SmilesGraphConverter:
    """This object handles the conversions betweens smiles and graphs"""

    @staticmethod
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
            x.append(node_matrix_map['atomic_num'].index(atom.GetAtomicNum()))
            x.append(node_matrix_map['chirality'].index(str(atom.GetChiralTag())))
            x.append(node_matrix_map['degree'].index(atom.GetTotalDegree()))
            x.append(node_matrix_map['formal_charge'].index(atom.GetFormalCharge()))
            x.append(node_matrix_map['num_hs'].index(atom.GetTotalNumHs()))
            x.append(node_matrix_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
            x.append(node_matrix_map['hybridization'].index(str(atom.GetHybridization())))
            x.append(node_matrix_map['is_aromatic'].index(atom.GetIsAromatic()))
            x.append(node_matrix_map['is_in_ring'].index(atom.IsInRing()))
            xs.append(x)

        x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e.append(edge_matrix_map['bond_type'].index(str(bond.GetBondType())))
            e.append(edge_matrix_map['stereo'].index(str(bond.GetStereo())))
            e.append(edge_matrix_map['is_conjugated'].index(bond.GetIsConjugated()))

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

        if edge_index.numel() > 0:  # Sort indices.
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


    @staticmethod
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
            atom.SetFormalCharge(node_matrix_map['formal_charge'][data.x[i, 3].item()])
            atom.SetNumExplicitHs(node_matrix_map['num_hs'][data.x[i, 4].item()])
            atom.SetNumRadicalElectrons(
                node_matrix_map['num_radical_electrons'][data.x[i, 5].item()])
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