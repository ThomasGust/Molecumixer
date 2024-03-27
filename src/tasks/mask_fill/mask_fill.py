from tasks import Task
import torch.nn as nn

class GraphMasker:
    """This function handles masking certain portions of a molecular graph"""

    def __init__(self):
        pass

class GraphMaskPredictionModel(nn.Module):
    """Given an encoder model, this module will reconstruct the masked portion of a graph from a latent vector"""

    def __init__(self):
        super(GraphMaskPredictionModel, self).__init__()

class GraphMaskTask(Task):
    """Implements the pretraining task for mask fill in or mask prediction"""

    def __init__(self):
        super().__init__("mask_prediction")