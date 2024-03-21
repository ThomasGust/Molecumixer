import sys
import torch
import numpy as np
import torch.nn.functional as F


class GraphPadding:
    """This is a utility module designed to pad a graph to a fixed size so other pretraining tasks can work better"""

    def __init__(self, max_molecule_size, max_edges, token=0):
        self.max_molecule_size = max_molecule_size
        self.max_edges = max_edges

        self.padding_token = token

    def pad_nodes(self, n):
        ln = n.size()[0]
        padding_length = self.max_molecule_size - ln

        return F.pad(input=n, pad=(0,0, 0, padding_length), mode='constant', value=self.padding_token)

    def pad_edge_attr(self,e):
        le = e.size()[0]
        padding_length = self.max_edges - le

        return F.pad(input=e, pad=(0, 0, 0, padding_length), mode='constant', value=self.padding_token)

    def pad_edge_index(self, e):
        le = e.size()[1]
        padding_length = self.max_edges - le
        
        return F.pad(input=e, pad=(0, padding_length, 0, 0), mode='constant', value=self.padding_token)

    def compute_mask(self, t):
        """This function computes a mask for where a tensor equals the padding token"""
        return t != self.padding_token

if __name__ == "__main__":
    pass