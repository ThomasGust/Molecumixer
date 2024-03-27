# TODO THIS IS THE LAST BIG FILE THAT STILL NEEDS TO START GETTING REFACTORED
# THE CURRENT ARCHITECTURE FOR TRAINING IS PROBABLY NOT THE BEST AND IT MIGHT BE A GOOD IDEA TO COMPUTE PRETRAINING TARGETS ON THE FLY INSTEAD OF SAVING THEM IN A DATALOADER   
import sys
sys.path.append('C:\\Users\\Thomas\\OneDrive\\Apps\\Documents\\GitHub\\Molecumixer\\src')
from utils import *
from config import *
from models import CGTNN
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils import pathjoin
import pickle as pkl

from tasks import save_task
from itertools import chain
#from config BEST_DEVICE

# THIS WHOLE FILE IS VERY UNORGANIZED AND NEEDS TO GET REDONE AT SOME POINT
print("FINISHED IMPORTS")

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

        total_parameters = self.get_params()
        #print(list(total_parameters))
        self.optimizer = OPTIM_DICT[optimizer](total_parameters, lr=init_lr)

        self.scheduler = SCHEDULER_DICT[scheduler](self.optimizer, mode='min', patience=self.scheduler_patience)

        self.encoder.to(BEST_DEVICE)
    
    def get_params(self):
        d = [{"params":self.encoder.parameters()}]

        for task in self.tasks:
            d.append({"params":task.model.parameters()})
    
        return d
    
    def step(self, batch):
        batch.to(BEST_DEVICE)

        latent = self.encoder(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        latent.to(BEST_DEVICE)

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
    
    def full_train(self):
        for e in tqdm(range(self.epochs)):
            self.train_epoch(e)

class Dojo:
    """This is the training environment in which our model will be pretrained. It does not expose any methods other than init. All training will happen through Sensei, this object merely wraps in hyperparameters."""
    #REMAKE COMMIT POWERs
    def __init__(self, log_sp, hyperparam_config_path):
        #TODO tasks will one day be added to the hyperparameter configuration

        self.logger_save_path = log_sp
        self.hyperparam_config_path = hyperparam_config_path

        self.hyperparams = load_dojo_config(hyperparam_config_path).as_dict()

        self.train_loader = torchload(pathjoin(self.hyperparams['dataloader_root'], "train_loader.moldata"))
        self.test_loader = torchload(pathjoin(self.hyperparams['dataloader_root'], "test_loader.moldata"))

        self.encoder = CGTNN(feature_size=9, embedding_size=self.hyperparams['model_embedding_size'],
                             attention_heads=self.hyperparams['model_attention_heads'],
                             n_layers=self.hyperparams['model_layers'],
                             dropout_ratio=self.hyperparams['model_dropout_rate'],
                             top_k_ratio=self.hyperparams['model_top_k_ratio'],
                             top_k_every_n=self.hyperparams['model_top_k_every_n'],
                             dense_neurons=self.hyperparams['model_dense_neurons'],
                             edge_dim=3)
        print("CREATED ENCODER")
        
        #TODO Task hyperparameters should also be saved to a config in the future
        self.tasks = [
            #ClusterPredictionTask(),
            DescriptorPredictionTask(self.hyperparams['model_embedding_size'], self.hyperparams['model_embedding_size']*2, output_dims=None, include_g3=True),
            FingerprintPredictionTask(self.hyperparams['model_embedding_size'], self.hyperparams['model_embedding_size']*2, [1024, 1024, 1024, 1024, 1024]),
            ShufflingPredictionTask(self.hyperparams['model_embedding_size'], self.hyperparams['model_embedding_size']*2, chunks=30, maximum_hamming_distance=3)
        ]
        for task in self.tasks:
            task.model.to(BEST_DEVICE)
        #self.tasks = [task.to(BEST_DEVICE) for task in self.tasks]

        self.log_names = [t.name for t in self.tasks]
        self.log_keys = []

        for name in self.log_names:
            self.log_keys.append(f"{name}_training_accuracy")
            self.log_keys.append(f"{name}_testing_accuracy")

            self.log_keys.append(f"{name}_training_loss")
            self.log_keys.append(f"{name}_testing_loss")
        
        self.logger = LogCallback(self.logger_save_path, keys=self.log_keys, tasks=self.tasks)
        print("CREATED LOGGER")

        self.sensei = Sensei(self.encoder, self.tasks, self.hyperparams['epochs'], batch_size=self.hyperparams['batch_size'], 
                             train_dataloader=self.train_loader, test_dataloader=self.test_loader, optimizer=self.hyperparams['optimizer'],
                             scheduler=self.hyperparams['scheduler'], scheduler_patience=self.hyperparams['scheduler_patience'], init_lr=self.hyperparams['learning_rate'],
                             log_callback=self.logger)
        
if __name__ == "__main__":
    dojo = Dojo(log_sp="logs", hyperparam_config_path="config.tc")
    dojo.sensei.full_train()