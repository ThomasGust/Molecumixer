import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\GitHub\\Molecumixer')
from config import (NUM_AROMATIC, NUM_ATOMS, NUM_BOND_TYPES, NUM_CHIRALITIES, NUM_CONJUGATED, NUM_DEGREES, NUM_FORMAL_CHARGES,
                    NUM_HS, NUM_INRING, NUM_RADICAL_ELECTRONS, NUM_HYBRIDIZATION, NUM_STEREO, MAX_MOLECULE_SIZE, E_MAP, ELEMENT_BASE, NODE_SHUFFLE_DECODER_DIMENSION)
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

# GVAE HAS BEEN REFACTORED TO THE MOLECULAR RECONSTRUCTION TASK, THERE IS STILL SOME CODE HERE THAT NEEDS TO BE MOVED WHICH IS WHY THIS FILE STILL EXISTS
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
    dataloader = torchload("data\\loaders\\sample_loader.moldata")

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

        p = f"checkpoints\\argvaet\\EPOCH_{epoch}"
        if os.path.exists(p):
            os.mkdir(p)
        torchdump(os.path.join(p, "argvaet.sd"), gvae)
 