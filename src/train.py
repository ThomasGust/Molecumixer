import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\GitHub\\Molecumixer\\src')
from utils import (torchload, dump, load, torchdump)
from utils import SUPPORTED_EDGES, SUPPORTED_ATOMS, MAX_MOLECULE_SIZE
from config import NODE_SHUFFLE_DECODER_DIMENSION, BEST_PARAMETERS
from models import CGTNN, LinearProjection, GVAE
from itertools import chain
import os
from utils import to_smiles, avg, is_rational, filter_inf, concat_generators, makeifnot, pathjoin, rmif

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
from orientations import permute_each_nodes, permute_nodes
from utils import pathjoin

# THIS WHOLE FILE IS VERY UNORGANIZED AND NEEDS TO GET REDONE AT SOME POINT
print("FINISHED IMPORTS")

BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
#device = torch.device("cpu")
dataloader = torchload("data\\loaders\\sample_loader.moldata")
print("LOADED DATALOADER")


# Mark raised a good point about how we should execute different pretraining tasks at an epoch by epoch level rather than a batch by batch level.
# We will have different pretraining "stages" which should also make the code a lot neater.

class PretrainingPhase:
    # This object will define a pretraining phase for our model, it will take a pretraining function which will be executed each epoch this pretraining phase is called.
    
    def __init__(self, model, epoch_function, epochs, optimizer, training_dataset, validation_dataset, lr_scheduler):
        self.epoch_function = epoch_function
        self.epochs = epochs
        
        self.model = model
        self.optimizer = optimizer
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.lr_scheduler = lr_scheduler


    def phase(self):
        history = self.epoch_function(self.model, self.optimizer, self.training_dataset, self.validation_dataset, self.lr_scheduler)
        return history

class ModelTrainer:
    
    def __init__(self, model, root_directory, phases: list[PretrainingPhase]):
        self.model = model
        self.root_directory = root_directory
        self.phases = phases

        #regenerate the root directory
        rmif(root_directory)
        makeifnot(root_directory)
    
    def execute_phase(self):
        pass

class LogCallback:
    pass

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

node_shuffle_projection = LinearProjection(BEST_PARAMETERS['model_embedding_size'][0], NODE_SHUFFLE_DECODER_DIMENSION).to(device)
    
blr = 0.001
encoder_optimizer = optim.Adam(model.parameters(), lr=blr)

params = concat_generators(model.parameters(),descriptor_proj.parameters(),descriptor3d_proj.parameters()
                             ,graph_descriptors_proj.parameters(),mfp2_proj.parameters(),mfp3_proj.parameters(),
                             maccs_proj.parameters(),rdkfp_proj.parameters(),avfp_proj.parameters())

total_optimizer = optim.Adam(params, lr=1e-3)
p = 3
scheduler = ReduceLROnPlateau(encoder_optimizer, 'min',patience=p, )

shuffle_params = concat_generators(model.parameters(), node_shuffle_projection.parameters())
shuffle_optimizer = optim.Adam(shuffle_params, lr=1e-3)
p=3
shuffle_scheduler = ReduceLROnPlateau(shuffle_optimizer, 'min', patience=p)

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

    _shuffle_losses = []

    for i, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        #print(batch.size)

        if i % 4 != 0:
            # I am currently unsure how I should handle training for multiple tasks, below is a descriptor and fingerprint training step
            # I will probably end up needing to design a router for this kind of task

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
        
        else:
            # Now we will actually train a shuffle prediction batch
            # Training is horrifically unorganized right now
            #nodes, orientation = permute_each_nodes(batch.cpu(), NODE_SHUFFLE_DECODER_DIMENSION, int(NODE_SHUFFLE_DECODER_DIMENSION/3))


            x = permute_each_nodes(batch.cpu(), NODE_SHUFFLE_DECODER_DIMENSION, int(NODE_SHUFFLE_DECODER_DIMENSION/3))
            nodes, orientation = x['x'], x['orientation']
            batch.to(device)

            nodes = torch.tensor(nodes)
            embedding = model(nodes.float().to(device), batch.edge_attr.float().to(device), batch.edge_index.to(device), batch.batch.to(device))
            #print(embedding.shape)
            
            node_shuffle_prediction = node_shuffle_projection(embedding)
            
            shuffle_label = torch.tensor(orientation).float().to(device)

            shuffle_loss = F.binary_cross_entropy_with_logits(node_shuffle_prediction.float(), shuffle_label.float())

            _shuffle_losses.append(shuffle_loss.item())

            #We cannot just use the same loss, I think that would really throw off the optimizer.
            # For now I am just using a seperate optimizer

            shuffle_optimizer.zero_grad()
            shuffle_loss.backward()
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
        train_one_epoch(epoch, model, dataloader, sp=f"checkpoints\\cgtnn\\EPOCH{epoch}", stats_sp=f"checkpoints\\cgtnn\\EPOCH{epoch}")
    data = load_stats("checkpoints\\cgtnn")
    dump("checkpoints\\cgtnn\\cgtnn.stats")
