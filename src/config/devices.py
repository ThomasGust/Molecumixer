import torch

BEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")