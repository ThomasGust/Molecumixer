import torch.nn as nn
import torch
import torch.nn.functional as F

def rmse(inputs, targets):
    return torch.sqrt(F.mse_loss(inputs, targets))

class DescriptorGenerator:
    """This object handles the generation of molecular descriptors"""

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




