import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\Chemix\\molecular_analysis')
from config import (NUM_AROMATIC, NUM_ATOMS, NUM_BOND_TYPES, NUM_CHIRALITIES, NUM_CONJUGATED, NUM_DEGREES, NUM_FORMAL_CHARGES,
                    NUM_HS, NUM_INRING, NUM_RADICAL_ELECTRONS, NUM_HYBRIDIZATION, NUM_STEREO, MAX_MOLECULE_SIZE, E_MAP, ELEMENT_BASE)
from paddings import pad_graph_batch, compute_mask
from utils import torchload, torchdump
from models import CGTNN
from config import MAX_EDGES
from torch import optim
from tqdm import tqdm
import os

ATOMIC_NUMBERS =  list(range(0, 119))
SUPPORTED_ATOMS = [ELEMENT_BASE[i][0] for i in ATOMIC_NUMBERS]
SUPPORTED_EDGES = E_MAP['bond_type']

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

def edge_list_to_adj_matrix(edge_list, num_nodes):
    # Create a zero matrix of shape [num_nodes, num_nodes]
    adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    adj_matrix[edge_list[0], edge_list[1]] = 1
    adj_matrix[edge_list[1], edge_list[0]] = 1  # Assuming undirected edges
    return adj_matrix



def adj_matrix_to_edge_list(adj_matrix):
    upper_triangle_indices = torch.triu_indices(MAX_MOLECULE_SIZE, MAX_MOLECULE_SIZE, offset=1, device=adj_matrix.device)
    edges = adj_matrix[i][upper_triangle_indices[0], upper_triangle_indices[1]]
    valid_edges = edges.nonzero(as_tuple=False).squeeze()

    src_nodes = upper_triangle_indices[0][valid_edges]
    tgt_nodes = upper_triangle_indices[1][valid_edges]
    
    edge_list = torch.stack([src_nodes, tgt_nodes], dim=1)
    
    return edge_list

def edge_index_loss(true_edge_index, adj_logits, batch_dim):
    adj_matrix = edge_list_to_adj_matrix(true_edge_index, MAX_MOLECULE_SIZE*batch_dim)
    loss = F.binary_cross_entropy_with_logits(adj_logits, adj_matrix)
    return loss

def kl_loss_fn(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reconstruction_loss_fn(node_truth, edge_truth, node_logits, edge_logits,
                        node_weights=[1.0 for _ in range(9)],
                        edge_weights=[1.0 for _ in range(3)],
                        node_weight=1.0,
                        edge_weight=1.0):

    atom_logits, chirality_logits, degree_logits, fc_logits, hs_logits, rad_electron_logits, hybridization_logits, aromatic_logits, ring_logits = node_logits
    edge_type_logits, stereo_logits, conjugated_logits = edge_logits

    node_mask = compute_mask(node_truth, -10)[:, 0]
    edge_mask = compute_mask(edge_truth, -10)[:, 0]

    node_truth = node_truth[node_mask]
    edge_truth = edge_truth[edge_mask]

    atom_truth = node_truth[:, 0]
    chirality_truth = node_truth[:, 1]
    degree_truth = node_truth[:, 2]
    fc_truth = node_truth[:, 3]
    hs_truth = node_truth[:, 4]
    rad_electron_truth = node_truth[:, 5]
    hybridization_truth = node_truth[:, 6]
    aromatic_truth = node_truth[:, 7]
    ring_truth = node_truth[:, 8]

    edge_type_truth = edge_truth[:, 0]
    stereo_truth = edge_truth[:, 1]
    conjugated_truth = edge_truth[:, 2]

    atom_loss = F.cross_entropy(atom_logits[node_mask], atom_truth)*node_weights[0]
    chirality_loss = F.cross_entropy(chirality_logits[node_mask], chirality_truth)*node_weights[1]
    degree_loss = F.cross_entropy(degree_logits[node_mask], degree_truth)*node_weights[2]
    fc_loss = F.cross_entropy(fc_logits[node_mask], fc_truth)*node_weights[3]
    hs_loss = F.cross_entropy(hs_logits[node_mask], hs_truth)*node_weights[4]
    rad_electron_loss = F.cross_entropy(rad_electron_logits[node_mask], rad_electron_truth)*node_weights[5]
    hybridization_loss = F.cross_entropy(hybridization_logits[node_mask], hybridization_truth)*node_weights[6]
    aromatic_loss = F.binary_cross_entropy_with_logits(aromatic_logits[node_mask], aromatic_truth.float())*node_weights[7]
    ring_loss = F.binary_cross_entropy_with_logits(ring_logits[node_mask], ring_truth.float())*node_weights[8]

    edge_type_loss = F.cross_entropy(edge_type_logits[edge_mask], edge_type_truth)*edge_weights[0]
    stereo_loss = F.cross_entropy(stereo_logits[edge_mask], stereo_truth)*edge_weights[1]
    conjugated_loss = F.binary_cross_entropy_with_logits(conjugated_logits[edge_mask], conjugated_truth.float())*edge_weights[2]

    rl_loss = node_weight*(atom_loss+chirality_loss+degree_loss+fc_loss+hs_loss+rad_electron_loss+hybridization_loss+aromatic_loss+ring_loss)+edge_weight*(edge_type_loss+stereo_loss+conjugated_loss)

    return rl_loss

def total_loss_fn(rl_loss, adj_loss, kl_loss, rl_lambda=1.0, adj_lambda=1.0, kl_lambda=1.0):
    return rl_lambda*rl_loss+adj_lambda*adj_loss+kl_lambda*kl_loss

if __name__ == "__main__":
    sample_data = torch.ones((32, 1024))
    device = torch.device("cpu")
    dataloader = torchload("molecular_analysis\\data_dir\\loaders\\sample_loader.moldata")

    BEST_PARAMETERS = {
    "batch_size": [128],
    "learning_rate": [0.01],
    "weight_decay": [0.0001],
    "sgd_momentum": [0.8],
    "scheduler_gamma": [0.8],
    "pos_weight": [1.3],
    "model_embedding_size": [1024],
    "model_attention_heads": [6],
    "model_layers": [8],
    "model_dropout_rate": [0.2],
    "model_top_k_ratio": [0.5],
    "model_top_k_every_n": [1],
    "model_dense_neurons": [256]
    }
    model = CGTNN(feature_size=9,
                embedding_size=BEST_PARAMETERS['model_embedding_size'][0],
                attention_heads=BEST_PARAMETERS['model_attention_heads'][0],
                n_layers=BEST_PARAMETERS['model_layers'][0],
                dropout_ratio=BEST_PARAMETERS['model_dropout_rate'][0],
                top_k_ratio=BEST_PARAMETERS['model_top_k_ratio'][0],
                top_k_every_n=BEST_PARAMETERS['model_top_k_every_n'][0],
                dense_neurons=BEST_PARAMETERS['model_dense_neurons'][0],
                edge_dim=3)

    #model.to(device)

    gvae = GVAE(feature_size=9,
                edge_dim=2,
                encoder_size=BEST_PARAMETERS['model_embedding_size'][0],
                latent_dim=BEST_PARAMETERS['model_embedding_size'][0],
                supported_edges=SUPPORTED_EDGES,
                supported_atoms=SUPPORTED_ATOMS,
                max_molecule_size=MAX_MOLECULE_SIZE,
                decoder_neurons=512)
    #gvae.to(device)
    optimizer = optim.Adam(gvae.parameters(), lr=1e-4)

    for epoch in range(300):
        print(f"EPOCH {epoch}")
        for i, sample in enumerate(pbar := tqdm(dataloader)):
            sample, batch_dim = pad_graph_batch(sample, -10, 32)

            #print(sample.x.size())
            if sample.x.size()[0] != 8000:
                pass
            else:
                optimizer.zero_grad()

                node_logits, edge_logits, edge_attr_logits, mu, logvar = gvae(sample_data.to(device), batch_dim)
                edge_type_logits, stereo_logits, conjugated_logits = edge_attr_logits
                atom_logits, chirality_logits, degree_logits, fc_logits, hs_logits, rad_electron_logits, hybridization_logits, aromatic_logits, ring_logits = node_logits

                rl_loss = reconstruction_loss_fn(sample.x, sample.edge_attr, node_logits, edge_attr_logits)
                kl_loss = kl_loss_fn(mu, logvar)
                adj_loss = edge_index_loss(sample.edge_index, edge_logits, batch_dim)
                total_loss = total_loss_fn(rl_loss, kl_loss, adj_loss, adj_lambda=10)

                total_loss.backward()
                #adj_loss.backward()
                optimizer.step()
                pbar.set_description(f"{total_loss.item()}, {adj_loss.item()*10}")

        p = f"molecular_analysis\\checkpoints\\argvaet\\EPOCH_{epoch}"
        if os.path.exists(p):
            os.mkdir(p)
        torchdump(os.path.join(p, "argvaet.sd"), gvae)
 