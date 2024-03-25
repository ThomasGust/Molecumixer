from tasks import Task

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit.Avalon import pyAvalonTools as avalon

class FingerprintCalculator:
    """
    This object is similar to our DescriptorCalculator, it will compute molecular fingerprints given any molecule
    """

    def __init__(self, fps=["mfp2", "mfp3", "maccs", "rdkfp", "avfp"]):
        self.fps = fps

        self.fp_d = {"mfp2":lambda mol: self.calculate_mfp(mol, 2),
                     "mfp3":lambda mol: self.calculate_mfp(mol, 3),
                     "maccs":lambda mol: self.calculate_maccs(mol),
                     "rdkfp":lambda mol: self.calculate_rdkfp(mol),
                     "avfp":lambda mol: self.calculate_avfp(mol)}
    @staticmethod
    def fp2np(fp):
        """Converts an rdkit fingerprint bit vector into a numpy array"""
        return np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0') 

    def calculate_mfp(self, mol, r):
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, r)
        return self.fp2np(mfp)
    
    def calculate_maccs(self, mol):
        maccs = MACCSkeys.GenMACCSKeys(mol)
        return self.fp2np(maccs)
    
    def calculate_rdkfp(self, mol):
        rdkfp = RDKFingerprint(mol)
        return self.fp2np(rdkfp)
    
    def calculate_avfp(self, mol):
        avfp = avalon.GetAvalonFP(mol)
        return self.fp2np(avfp)
    
    def calculate_fingerprints(self, smiles):
        #TODO I should add support for both a mol and a pre-embedded RdKit Molecule
        o_d = {}
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        for fp_type in self.fps:
            computer = self.fp_d[fp_type]
            fp = computer(mol)

            o_d[fp_type] = fp
        
        return o_d

class FingerprintPredictionModel:
    """This module will learn to predict molecular fingerprints from a latent vector"""

    def __init__(self, encoder_dim, hidden_dim, fingerprint_dims, activation=F.relu):
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.fingerprint_dims = fingerprint_dims

        self.ls = []
        self.fos = []

        for dim in self.fingerprint_dims:
            self.ls.append(nn.Linear(self.encoder_dim, self.hidden_dim))
            self.fos.append(nn.Linear(self.hidden_dim, dim))
        
        self.activation = activation

    
    def forward(self, x):
        xs = []

        for i, _ in enumerate(self.ls):
            _x = self.ls[i](x)
            _x = self.activation(_x)
            _x = self.fos[i](_x)
        
        return xs

    def compute_loss(self, logits, labels):
        """
        An equal weight BCE loss for any binary fingerprints this model might try to predict
        """
        losses = []

        for i, dim in enumerate(self.fingerprint_dims):
            l = F.binary_cross_entropy_with_logits(logits[i], labels[i])
            l /= dim
        
        return sum(losses)

class FingerprintPredictionTask(Task):
    """Implements the pre-training task for molecular fingerprint prediction"""

    def __init__(self, encoder_dim, hidden_dim, fingerprint_dims, activation=F.relu, fps=["mfp2", "mfp3", "maccs", "rdkfp", "avfp"]):
        super().__init__("fingerprint_prediction")

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.fingerprint_dims = fingerprint_dims
        self.activation = activation
        self.fps = fps
        
        self.model = FingerprintPredictionModel(encoder_dim=self.encoder_dim, hidden_dim=self.hidden_dim, fingerprint_dims=self.fingerprint_dims, activation=self.activation)
        self.fingerprint_calculator = FingerprintCalculator(self.fps)
    
    def task_step(self, encoder, batch):
        fingerprints = self.fingerprint_calculator.calculate_fingerprints(batch.smiles) # TODO I don't think any of the calculators are set up to handle batches
        
        latent = encoder(batch) # TODO Once more, I don't think this is the correct way to call the encoder
        pred = self.model(latent)

        loss = self.model.compute_loss(pred, fingerprints)

        return {"loss":loss, "pred":pred}

if __name__ == "__main__":
    task = FingerprintPredictionTask(encoder_dim=512, hidden_dim=1024, fingerprint_dims=["IDK"])
    print("LOADED TASK")