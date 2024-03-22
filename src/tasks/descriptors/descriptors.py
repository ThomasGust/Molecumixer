from tasks import Task

import torch.nn as nn
import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D, GraphDescriptors

def rmse(inputs, targets):
    return torch.sqrt(F.mse_loss(inputs, targets))

class DescriptorCalculator:
    """This object will be responsible for calculating all of our different molecular descriptors"""

    def __init__(self, include_g3=True):
        """
        include g3 will include 'non normal' descriptors in the final calculation 
        """
        self.include_g3 = include_g3

    def calculate_rdmol_descriptors(self, mol):
        return Descriptors.CalcMolDescriptors(mol)
    
    def calculate_graph_descriptors(self, mol):
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

    def calculate_3d_descriptors(self, mol):
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

    def calculate_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        rddescriptors = self.calculate_rdmol_descriptors(mol)
        if self.include_g3:
            descriptors_3d = self.calculate_3d_descriptors(mol)
            graph_descriptors = self.calculate_graph_descriptors(mol)

            descriptors = {"rd": rddescriptors, "3d":descriptors_3d, "graph":graph_descriptors}
        else:
            descriptors = {"rd": rddescriptors}
    
        return descriptors

class DescriptorPredictionModel:
    """This module will predict the molecular descriptors of a given molecule given a latent vector from an encoder"""

    def __init__(self, encoder_dim, hidden_dim, output_dims, activation=F.relu):

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        
        self.d_l1 = nn.Linear(self.encoder_dim, self.hidden_dim) # Descriptor Linear 1
        self.d_o = nn.Linear(self.hidden_dim, self.output_dims[0])

        self.d3_l1 = nn.Linear(self.encoder_dim, self.hidden_dim) # Descriptor 3D Linear 1
        self.d3_o = nn.Linear(self.hidden_dim, self.output_dims[1])

        self.dg_l1 = nn.Linear(self.encoder_dim, self.hidden_dim) # Graph Descriptors Linear 1
        self.dg_o = nn.Linear(self.hidden_dim, self.output_dims[2])

        self.activation = activation
    
    def forward(self, x):
        dx = self.d_l1(x)
        dx = self.activation(dx)
        dx = self.d_o(dx)

        d3x = self.d3_l1(x)
        d3x = self.activation(d3x)
        d3x = self.d3_o(d3x)

        dgx = self.dg_l1(x)
        dgx = self.activation(dgx)
        dgx = self.d3_o(dgx)

        return dx, d3x, dgx

    def compute_loss(self, logits, labels):
        """
        Computes an equal weight rmse loss for each of the different types of descriptors we are considering
        """
        dpred, d3pred, dgpred = logits
        d, d3, dg = labels

        dloss =  rmse(dpred, d)/self.output_dims[0]
        d3loss = rmse(d3pred, d3)/self.output_dims[1]
        dgloss = rmse(dgpred, dg)/self.output_dims[2]

        return dloss+d3loss+dgloss

class DescriptorPredictionTask(Task):
    "Implements the pre training task of molecular descriptor prediction"

    def __init__(self):
        super().__init__()
    
if __name__ == "__main__":
    descriptor_pred_model = DescriptorPredictionModel(512, 1024, [209, 9, 19])