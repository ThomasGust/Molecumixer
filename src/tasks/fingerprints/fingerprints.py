import torch
import torch.nn as nn
import torch.nn.functional as F

class FingerprintGenerator:
    """This object handles the generation of molecular fingerprints"""

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
