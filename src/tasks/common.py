import torch
import os

class Task:
    """This object will be the base class for all other pretraining tasks"""

    def __init__(self, name):
        self.name = name
    
    def task_step(self, batch):
        pass

def save_task(task, sp_root):
    torch.save(task.model.state_dict(), os.path.join(sp_root, f"{task.name}.sd"))
