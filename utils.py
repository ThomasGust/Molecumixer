import pickle as pkl
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.utils import to_dense_adj
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import ELEMENT_BASE as element_base
from config import MAX_MOLECULE_SIZE
from config import E_MAP

ATOMIC_NUMBERS =  list(range(0, 119))
SUPPORTED_ATOMS = [element_base[i][0] for i in ATOMIC_NUMBERS]
SUPPORTED_EDGES = E_MAP['bond_type']
DISABLE_RDKIT_WARNINGS = False

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

# A LOT OF THE FUNCTIONS AFTER THIS POINT WILL BE FROM THE DEEPFINDR SERIES: