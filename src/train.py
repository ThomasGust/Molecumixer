# TODO THIS IS THE LAST BIG FILE THAT STILL NEEDS TO START GETTING REFACTORED
# THE CURRENT ARCHITECTURE FOR TRAINING IS PROBABLY NOT THE BEST AND IT MIGHT BE A GOOD IDEA TO COMPUTE PRETRAINING TARGETS ON THE FLY INSTEAD OF SAVING THEM IN A DATALOADER   
import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\GitHub\\Molecumixer\\src')
from utils import (torchload, dump, load, torchdump)
from utils import SUPPORTED_EDGES, SUPPORTED_ATOMS, MAX_MOLECULE_SIZE
from config import *
from models import CGTNN, LinearProjection, GVAE
from itertools import chain
import os
from utils import to_smiles, avg, is_rational, filter_inf, concat_generators, makeifnot, pathjoin, rmif

import matplotlib.pyplot as plt
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
import pickle as pkl

from tasks import save_task

# THIS WHOLE FILE IS VERY UNORGANIZED AND NEEDS TO GET REDONE AT SOME POINT
print("FINISHED IMPORTS")

BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
#device = torch.device("cpu")
dataloader = torchload("data\\loaders\\sample_loader.moldata")
print("LOADED DATALOADER")

class LogCallback:
    """Handles all of the logging during pretraining"""

    def __init__(self, save_path, keys, tasks):
        self.save_path = save_path
        self.tasks = tasks
        self.keys = keys

        rmif(save_path)
        makeifnot(save_path)

        self.memory = {}

        for key in self.keys:
            self.memory[key] = []
        self.epoch = 0
        
        self.encoder = None
    
    def register(self, epoch_data, encoder, tasks):
        """Given the data gathered batchwise over one epoch this model will compute the epochwise average. This model will also save the encoder to a specified save directory"""

        for key in self.keys:
            data = epoch_data[key]
            avg = sum(data)/len(data)
            self.memory[key].append(avg)

        self.epoch += 1
        self.encoder = encoder
        self.tasks = tasks
    
    def save_memory(self):
        
        for key in self.keys:
            img_path = os.path.join(self.save_path, f"{key}.png")

            plt.plot(self.memory[key], len(list(self.memory)))
            plt.ylabel(key)
            plt.xlabel("Epochs")
            plt.legend()
            plt.savefig(img_path)
            plt.close()

        epoch_path = pathjoin(self.save_path, self.epoch)

        makeifnot(epoch_path)

        encoder_path = pathjoin(epoch_path, "encoder.sd")
        torch.save(self.encoder.state_dict(), encoder_path)

        for task in self.tasks:
            save_task(task, epoch_path)

        hist_path = os.path.join(self.save_path, "hist.pkl")

        with open(hist_path, "wb") as f:
            pkl.dump(self.memory, f)

class Sensei:
    """This object is responsible for actually teaching our model, it fits in nicely into the dojo object which stores the training ENVIRONMENT for our model"""

    def __init__(self, encoder, tasks, epochs, batch_size, train_dataloader, test_dataloader, optimizer, scheduler, scheduler_patience, init_lr, log_callback: LogCallback):
        self.encoder = encoder
        self.tasks = tasks
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.log_callback = log_callback
        self.init_lr = init_lr
        self.scheduler_patience = scheduler_patience

        total_parameters = concat_generators(self.encoder.parameters(), self.get_task_params())
        self.optimizer = OPTIM_DICT[optimizer](total_parameters, lr=init_lr)

        self.scheduler = SCHEDULER_DICT[scheduler](self.optimizer, mode='min', patience=self.scheduler_patience)
    
    def get_task_params(self):
        params = concat_generators([task.model.parameters() for task in self.tasks])
    
        return params
    
    def step(self, batch):
        latent = self.encoder(batch.x.float().to(device), batch.edge_attr.float().to(device), batch.edge_index.to(device), batch.batch.to(device))

        losses = {}
        for task in self.tasks:
            #TODO, right now, we are just considering, loss, in the future it would be great if we could also track some kind of accuracy
            task_d = task.task_step(latent, batch)
            task_loss = task_d['loss']
            losses[task.name] = task_loss

        t_losses = torch.tensor(list(losses.values())) # TODO I don't think this is recommended so I should probably find a better way to do this
        combined_loss = torch.mean(t_losses) # TODO This is worth reviewing, I'm not sure if taking the mean of the losses is the best way to go about this
        return combined_loss, losses
    
    def train_batch(self, batch):
        self.optimizer.zero_grad()
        combined_loss, losses = self.step(batch)

        combined_loss.backward()
        self.optimizer.step()

        return losses

    def test_batch(self, batch):
        combined_loss, losses = self.step(batch)
        return losses
    
    def train_epoch(self, e):
        epoch_data = {}

        for batch in tqdm(self.train_dataloader, desc=f"Training Epoch {e}"):
            losses = self.train_batch(batch)
            
            for key in list(losses.keys()):
                epoch_data[f"{key}_training_loss"] = losses[key]
        
        for batch in tqdm(self.test_dataloader, desc=f"Testing Epoch {e}"):
            losses = self.test_batch(batch)

            for key in list(losses.keys()):
                epoch_data[f"{key}_testing_loss"] = losses[key]
        
        self.log_callback.register(epoch_data, self.encoder, self.tasks)
        self.log_callback.save_memory()

class Dojo:
    """This is the training environment in which our model will be pretrained"""

    def __init__(self, tasks, log_sp, hyperparam_config_path):
        #TODO tasks will one day be added to the hyperparameter configuration

        self.logger_save_path = log_sp
        self.tasks = tasks
        self.hyperparam_config_path = hyperparam_config_path

        self.log_names = [t.name for t in self.tasks]
        self.log_keys = []

        for name in self.log_names:
            self.log_keys.append(f"{name}_training_accuracy")
            self.log_keys.append(f"{name}_testing_accuracy")

            self.log_keys.append(f"{name}_training_loss")
            self.log_keys.append(f"{name}_testing_loss")
        
        self.logger = LogCallback(self.logger_save_path, self.log_keys)

        self.hyperparams = load_dojo_config(hyperparam_config_path)

        self.train_loader = torchload(pathjoin(self.hyperparams['dataloader_root'], "train_loader.moldata"))
        self.test_loader = torchload(pathjoin(self.hyperparams['dataloader_root'], "test_loader.moldata"))

        self.encoder = CGTNN(feature_size=9, embedding_size=self.hyperparams['model_embedding_size'],
                             attention_heads=self.hyperparams['model_attention_heads'],
                             n_layers=self.hyperparams['model_layers'],
                             dropout_ratio=self.hyperparams['model_dropout_rate'],
                             top_k_ratio=self.hyperparams['model_top_k_ratio'],
                             top_k_every_n=self.hyperparams('model_top_k_every_n'),
                             dense_neurons=self.hyperparams['model_dense_neurons'],
                             edge_dim=9)
        
        self.sensei = Sensei(self.encoder, self.tasks, self.hyperparam_config_path['epochs'], batch_size=self.hyperparam_config_path['batch_size'], 
                             train_dataloader=self.train_loader, test_dataloader=self.test_loader, optimizer=self.hyperparams['optimizer'],
                             scheduler=self.hyperparams['scheduler'], scheduler_patience=self.hyperparams['scheduler_patience'], init_lr=self.hyperparams['learning_rate'],
                             log_callback=self.logger)