from tasks import Task

import torch
from torch.nn import Linear
import torch.nn.functional as F
from config import (NUM_AROMATIC, NUM_ATOMS, NUM_BOND_TYPES, NUM_CHIRALITIES, NUM_CONJUGATED, NUM_DEGREES, NUM_FORMAL_CHARGES,
                    NUM_HS, NUM_INRING, NUM_RADICAL_ELECTRONS, NUM_HYBRIDIZATION, NUM_STEREO, MAX_MOLECULE_SIZE, E_MAP, ELEMENT_BASE, NODE_SHUFFLE_DECODER_DIMENSION, MAX_EDGES)
import torch.nn as nn

class GVAE(nn.Module):

    def __init__(self, feature_size, edge_dim, encoder_size, latent_dim, supported_edges,
                  supported_atoms, max_molecule_size, decoder_neurons=512, node_embedding_dim=1):
        super(GVAE, self).__init__()
        self.feature_size = feature_size
        self.edge_dim = edge_dim
        #self.encoder = encoder
        self.encoder_size = encoder_size
        self.latent_dim = latent_dim
        self.supported_edges = supported_edges
        self.supported_atoms = supported_atoms
        self.max_molecule_size = max_molecule_size
        self.decoder_neurons = decoder_neurons
        self.node_embedding_dim = node_embedding_dim

        self.num_edge_types = len(supported_edges)
        self.num_atom_types = len(supported_atoms)


        self.mu_l = Linear(self.encoder_size, self.latent_dim)
        self.logvar_l = Linear(self.encoder_size, self.latent_dim)

        # The first two are the shared decoder layers, the rest are projection heads
        self.dl1 = Linear(self.latent_dim, self.decoder_neurons)
        self.dl2 = Linear(self.decoder_neurons, self.decoder_neurons)

        # This first set of layers is to decode the original node matrix
        atom_output_dim = MAX_MOLECULE_SIZE*NUM_ATOMS
        self.atom_decode = Linear(self.decoder_neurons, atom_output_dim)

        chirality_output_dim = MAX_MOLECULE_SIZE*NUM_CHIRALITIES
        self.chirality_decode = Linear(self.decoder_neurons, chirality_output_dim)

        degree_output_dim =  MAX_MOLECULE_SIZE*NUM_DEGREES
        self.degree_decode = Linear(self.decoder_neurons, degree_output_dim)

        formal_charge_dim = MAX_MOLECULE_SIZE*NUM_FORMAL_CHARGES
        self.formal_charge_decode = Linear(self.decoder_neurons, formal_charge_dim)

        num_hs_dim = MAX_MOLECULE_SIZE*NUM_HS
        self.num_hs_decode = Linear(self.decoder_neurons, num_hs_dim)

        num_radical_electrons_dim = MAX_MOLECULE_SIZE*NUM_RADICAL_ELECTRONS
        self.num_radical_electrons_decode = Linear(self.decoder_neurons, num_radical_electrons_dim)

        hybridization_dim = MAX_MOLECULE_SIZE * NUM_HYBRIDIZATION
        self.hybriziation_decode = Linear(self.decoder_neurons,hybridization_dim)

        # These are binary tasks, so the output dim can just be MAX_MOLECULE SIZE
        aromatic_dim = MAX_MOLECULE_SIZE
        self.aromatic_decode = Linear(self.decoder_neurons, aromatic_dim)

        ring_dim = MAX_MOLECULE_SIZE
        self.ring_decode = Linear(self.decoder_neurons, ring_dim)

        # This next set of layers has to do with predicting information about the edge matrix
        # Currently, I am unsure if I am calculating the edge output dimensions correctly, so I will need to
        # look at this if stuff goes wrong

        edge_output_dim = MAX_EDGES*NUM_BOND_TYPES
        self.edge_decode = Linear(self.decoder_neurons, edge_output_dim)

        stereo_output_dim = MAX_EDGES*NUM_STEREO
        self.stereo_decode = Linear(self.decoder_neurons, stereo_output_dim)

        conjugated_output_dim = MAX_EDGES
        self.conjugated_decode = Linear(self.decoder_neurons, conjugated_output_dim)

        # Now for the edge index decoder
        self.edge_index_decode = Linear(self.decoder_neurons, MAX_MOLECULE_SIZE*self.node_embedding_dim)

        node_shuffle_decode_dim = NODE_SHUFFLE_DECODER_DIMENSION
        self.node_shuffle_decoder = Linear(self.decoder_neurons, node_shuffle_decode_dim)

    
    def reparameterize(self, mu, logvar):
        if self.training:
            std_dev = torch.exp(logvar)
 
            r = torch.randn_like(std_dev)

            return r.mul(std_dev).add_(mu)
        else:
            return mu
    
    def decode_node_attrs(self, z, batch_dim):
        d = batch_dim * MAX_MOLECULE_SIZE

        atom_logits = self.atom_decode(z)
        atom_logits = atom_logits.view(d, NUM_ATOMS)
        atom_logits = F.softmax(atom_logits, 1)

        chirality_logits = self.chirality_decode(z)
        chirality_logits = chirality_logits.view(d, NUM_CHIRALITIES)
        chirality_logits = F.softmax(chirality_logits, 1)

        degree_logits = self.degree_decode(z)
        degree_logits = degree_logits.view(d, NUM_DEGREES)
        degree_logits = F.softmax(degree_logits, 1)

        fc_logits = self.formal_charge_decode(z)
        fc_logits = fc_logits.view(d, NUM_FORMAL_CHARGES)
        fc_logits = F.softmax(degree_logits, 1)

        num_hs_logits = self.num_hs_decode(z)
        num_hs_logits = num_hs_logits.view(d, NUM_HS)
        num_hs_logits = F.softmax(num_hs_logits, 1)

        num_rad_electrons_logits = self.num_radical_electrons_decode(z)
        num_rad_electrons_logits = num_rad_electrons_logits.view(d, NUM_RADICAL_ELECTRONS)
        num_rad_electrons_logits = F.softmax(num_rad_electrons_logits, 1)

        hybridization_logits = self.hybriziation_decode(z)
        hybridization_logits = hybridization_logits.view(d, NUM_HYBRIDIZATION)
        hybridization_logits = F.softmax(hybridization_logits, 1)

        aromatic_logits = self.aromatic_decode(z)
        aromatic_logits = aromatic_logits.view(d)
        aromatic_logits = F.sigmoid(aromatic_logits)

        ring_logits = self.ring_decode(z)
        #print(ring_logits.size())
        ring_logits = ring_logits.view(d)
        ring_logits = F.sigmoid(ring_logits)

        return (atom_logits, chirality_logits, degree_logits, fc_logits, num_hs_logits, num_rad_electrons_logits,
                hybridization_logits, aromatic_logits, ring_logits)
    
    def decode_edge_attrs(self, z, batch_dim):
        #type_logits = F.softmax(self.edge_decode(z), 1)
        d = batch_dim * MAX_EDGES

        type_logits = self.edge_decode(z)
        type_logits = type_logits.view(d, NUM_BOND_TYPES)
        type_logits = F.softmax(type_logits, 1)

        stereo_logits = self.stereo_decode(z)
        stereo_logits = stereo_logits.view(d, NUM_STEREO)
        stereo_logits = F.softmax(stereo_logits, 1)

        conjugated_logits = self.conjugated_decode(z)
        conjugated_logits = conjugated_logits.view(d)
        conjugated_logits = F.sigmoid(conjugated_logits)

        return (type_logits, stereo_logits, conjugated_logits)

    def decode_adjacency_matrix(self, z, batch_dim):
        e = self.edge_index_decode(z)
        e = e.view(batch_dim*MAX_MOLECULE_SIZE, self.node_embedding_dim)
        
        a = torch.mm(e, e.t())
        return F.sigmoid(a)

    def decode_graph(self, graph_z, batch_dim):
        z = F.relu(self.dl1(graph_z))
        z = F.relu(self.dl2(z))

        node_logits = self.decode_node_attrs(z, batch_dim)
        edge_attr_logits = self.decode_edge_attrs(z, batch_dim)
        edge_index_logits = self.decode_adjacency_matrix(z, batch_dim)

        return node_logits, edge_index_logits, edge_attr_logits

    def forward(self, latent, batch_dim):
        mu = self.mu_l(latent)
        logvar = self.logvar_l(latent)

        z = self.reparameterize(mu, logvar)

        node_logits, edge_index_logits, edge_attr_logits = self.decode_graph(z, batch_dim)

        return node_logits, edge_index_logits, edge_attr_logits, mu, logvar
 
class GraphDiscriminator:
    """Given an encoder model, this object will try to predict whether or not a molecular graph is real or generated through GVAE"""

class GraphReconstructionTask(Task):
    """Implements the graph reconstruction task"""

    def __init__(self):
        super().__init__()