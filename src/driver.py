#THIS IS A TEST FILE DUE TO THE WAY RELATIVE PYTHON PATHING WORKS
from torch.utils.data import DataLoader
from data.data import DataManager, MolecumixerDataset
import time

dm = DataManager("data\\build")
dm.merge()
dm.create_dataloader()
dataset = MolecumixerDataset("data\\build")
print("INITIALIZED DATASET")
loader = DataLoader(dataset, batch_size=32)
print("INITIALIZED DATALOADER")
print(len(loader))


for batch in loader:
    print(batch)
    print(batch.shape)
    break

