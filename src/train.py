import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\Visual studio code projects\\Chemix\\molecular_analysis')
from utils import (torchload, dump, load, torchdump)
from utils import SUPPORTED_EDGES, SUPPORTED_ATOMS, MAX_MOLECULE_SIZE
from models import CGTNN, LinearProjection, GVAE
from itertools import chain
import os
from data import to_smiles

import numpy as np
import torch_geometric
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from plotter import load_stats

print("FINISHED IMPORTS")

BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
#device = torch.device("cpu")
dataloader = torchload("molecular_analysis\\data_dir\\loaders\\sample_loader.moldata")
print("LOADED DATALOADER")

def avg(l):
    return sum(l)/len(l)

def filter_inf(t, nan=0.0, pinf=0.0, ninf=0.0):
    return torch.nan_to_num(t, nan=nan, posinf=pinf, neginf=ninf)

def is_rational(mol_g):
    try:
        _ = to_smiles(mol_g)
        return True
    except Exception:
        return False

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

model.to(device)

descriptor_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=209).to(device)
descriptor3d_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=9).to(device)
graph_descriptors_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=19).to(device)

mfp2_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=2048).to(device)
mfp3_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=2048).to(device)
maccs_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=167).to(device)
rdkfp_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=2048).to(device)
avfp_proj = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], n_o=512).to(device)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
blr = 0.001
encoder_optimizer = optim.Adam(model.parameters(), lr=blr)

def concat_generators(*args):
      for gen in args:
          yield from gen

params = concat_generators(model.parameters(),descriptor_proj.parameters(),descriptor3d_proj.parameters()
                             ,graph_descriptors_proj.parameters(),mfp2_proj.parameters(),mfp3_proj.parameters(),
                             maccs_proj.parameters(),rdkfp_proj.parameters(),avfp_proj.parameters())

total_optimizer = optim.Adam(params, lr=1e-3)

p = 3
scheduler = ReduceLROnPlateau(encoder_optimizer, 'min',patience=p, )

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, sp=None, stats_sp=None):

    _descriptor_losses = []
    _descriptors3d_losses = []
    _graph_descriptors_losses = []

    _accum_descriptor_losses = []

    _mfp2_losses = []
    _mfp3_losses = []
    _maccs_losses = []
    _rdkfp_losses = []
    _avfp_losses = []

    _fp_losses = []

    _t_losses = []

    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        #print(batch.size)
        descriptors = filter_inf(torch.tensor(np.array(batch.descriptors)).to(device).float())
        descriptors3d = filter_inf(torch.tensor(np.array(batch.descriptors3d)).to(device).float())
        graph_descriptors = filter_inf(torch.tensor(np.array(batch.graph_descriptors)).to(device).float())

        mfp2 = torch.tensor(np.array(batch.mfp2)).to(device)#.float()
        mfp3 = torch.tensor(np.array(batch.mfp3)).to(device).float()
        maccs = torch.tensor(np.array(batch.maccs)).to(device).float()
        rdkfp = torch.tensor(np.array(batch.rdkfp)).to(device).float()
        avfp = torch.tensor(np.array(batch.avfp)).to(device).float()

        embedding = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

        descriptor_pred = descriptor_proj(embedding)
        descriptor3d_pred = descriptor3d_proj(embedding)
        graph_descriptor_pred = graph_descriptors_proj(embedding)

        mfp2_pred = mfp2_proj(embedding)
        mfp3_pred = mfp3_proj(embedding)
        maccs_pred = maccs_proj(embedding)
        rdkfp_pred = rdkfp_proj(embedding)
        avfp_pred = avfp_proj(embedding)

        descriptor_loss = torch.sqrt(F.mse_loss(descriptor_pred.float(), descriptors.float()))
        descriptors3d_loss = torch.sqrt(F.mse_loss(descriptor3d_pred.float(), descriptors3d.float()))
        graph_descriptor_loss = torch.sqrt(F.mse_loss(graph_descriptor_pred.float(), graph_descriptors.float()))

        mfp2_loss = F.binary_cross_entropy_with_logits(mfp2_pred.float(), mfp2.float())
        mfp3_loss = F.binary_cross_entropy_with_logits(mfp3_pred.float(), mfp3.float())
        maccs_loss = F.binary_cross_entropy_with_logits(maccs_pred.float(), maccs.float())
        rdkfp_loss = F.binary_cross_entropy_with_logits(rdkfp_pred.float(), rdkfp.float())
        avfp_loss = F.binary_cross_entropy_with_logits(avfp_pred.float(), avfp.float())

        _descriptor_losses.append(descriptor_loss.item())
        _descriptors3d_losses.append(descriptors3d_loss.item())
        _graph_descriptors_losses.append(graph_descriptor_loss.item())
        _accum_descriptor_losses.append(descriptor_loss.item()+descriptors3d_loss.item()+graph_descriptor_loss.item())

        _mfp2_losses.append(mfp2_loss.item())
        _mfp3_losses.append(mfp3_loss.item())
        _maccs_losses.append(maccs_loss.item())
        _rdkfp_losses.append(rdkfp_loss.item())
        _avfp_losses.append(avfp_loss.item())
        _fp_losses.append(mfp2_loss.item()+mfp3_loss.item()+maccs_loss.item()+rdkfp_loss.item()+avfp_loss.item())

        t_loss = mfp2_loss+mfp3_loss+maccs_loss+rdkfp_loss+avfp_loss+descriptor_loss+descriptors3d_loss+graph_descriptor_loss
        _t_losses.append(t_loss.item())
        total_optimizer.zero_grad()
        t_loss.backward()
        total_optimizer.step()

    epoch_descriptor_loss = avg(_descriptor_losses)
    epoch_descriptor3d_loss = avg(_descriptors3d_losses)
    epoch_graph_descriptors_loss = avg(_graph_descriptors_losses)

    epoch_accum_descriptor_loss = avg(_accum_descriptor_losses)

    epoch_mfp2_loss = avg(_mfp2_losses)
    epoch_mfp3_loss = avg(_mfp3_losses)
    epoch_maccs_loss = avg(_maccs_losses)
    epoch_rdkfp_loss = avg(_rdkfp_losses)
    epoch_avfp_loss = avg(_avfp_losses)

    epoch_fp_loss = avg(_fp_losses)

    epoch_loss = avg(_t_losses)

    scheduler.step(epoch_loss, epoch=epoch)

    if sp is not None:
        if not os.path.exists(sp):
            os.mkdir(sp)
        torch.save(model.state_dict(), os.path.join(sp, "encoder.sd"))

        torch.save(descriptor_proj.state_dict(), os.path.join(sp, "descriptorproj.sd"))
        torch.save(descriptor3d_proj.state_dict(), os.path.join(sp, "descriptor3dproj.sd"))
        torch.save(graph_descriptors_proj.state_dict(), os.path.join(sp, "graphdescriptorproj.sd"))

        torch.save(mfp2_proj.state_dict(), os.path.join(sp, "mfp2proj.sd"))
        torch.save(mfp3_proj.state_dict(), os.path.join(sp, "mfp3proj.sd"))
        torch.save(maccs_proj.state_dict(), os.path.join(sp, "maccsproj.sd"))
        torch.save(rdkfp_proj.state_dict(), os.path.join(sp, "rdkfpproj.sd"))
        torch.save(avfp_proj.state_dict(), os.path.join(sp, "avfpproj.sd"))
    
    current_lr = encoder_optimizer.param_groups[0]['lr']

    if stats_sp is not None:
        if not os.path.exists(stats_sp):
            os.mkdir(stats_sp)

        dump(os.path.join(stats_sp, f"stats_epoch_{epoch}.stats"), {
            'descriptor loss': epoch_descriptor_loss,
            '3d descriptor loss': epoch_descriptor3d_loss,
            'graph descriptor loss': graph_descriptor_loss,
            'accumulated descriptor loss': epoch_accum_descriptor_loss,
            'mfp2 loss': epoch_mfp2_loss,
            'mfp3 loss': epoch_mfp3_loss,
            'maccs loss': epoch_maccs_loss,
            'rdkfp loss': epoch_rdkfp_loss,
            'avfp loss': epoch_avfp_loss,
            'fingerprint loss': epoch_fp_loss,
            'total loss': epoch_loss,
            'lr': current_lr
        })

    print(f"EPOCH: {epoch}")
    print(f"LEARNING RATE: {current_lr}")
    print()
    print(f"DESCRIPTOR LOSS: {epoch_descriptor_loss}")
    print(f"3D DESCRIPTOR LOSS: {epoch_descriptor3d_loss}")
    print(f"GRAPH DESCRIPTOR LOSS: {epoch_graph_descriptors_loss}")
    print(f"ACCUMULATED DESCRIPTOR LOSS: {epoch_accum_descriptor_loss}")
    print()
    print(f"MFP2 LOSS: {epoch_mfp2_loss}")
    print(f"MFP3 LOSS: {epoch_mfp3_loss}")
    print(f"MACCS LOSS: {epoch_maccs_loss}")
    print(f"RDKFP LOSS: {epoch_rdkfp_loss}")
    print(f"AVFP LOSS: {epoch_avfp_loss}")
    print(f"FINGERPRINT LOSS: {epoch_fp_loss}")
    print()
    print(f"TOTAL LOSS: {epoch_loss}")

if __name__ == "__main__":
    for epoch in range(50):
        train_one_epoch(epoch, model, dataloader, sp=f"molecular_analysis\\checkpoints\\cgtnn\\EPOCH{epoch}", stats_sp=f"molecular_analysis\\checkpoints\\cgtnn\\EPOCH{epoch}")
    data = load_stats("molecular_analysis\\checkpoints\\cgtnn")
    dump("molecular_analysis\\checkpoints\\cgtnn\\cgtnn.stats")
