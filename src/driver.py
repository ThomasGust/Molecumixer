#THIS IS A TEST FILE DUE TO THE WAY RELATIVE PYTHON PATHING WORKS
from torch.utils.data import DataLoader
from data.data import DataManager, MolecumixerDataset, DatasetBuilder, DescriptorCalculator, GraphCalculator
import time
import torch
"""
dm = DataManager("data\\build")
dm.merge()
dm.create_dataloader()
dataset = MolecumixerDataset("data\\build")
print("INITIALIZED DATASET")
loader = DataLoader(dataset, batch_size=32)
torch.save(loader, "loader.pt")
print("INITIALIZED DATALOADER")
print(len(loader))


for batch in loader:
    print(batch)
    print(list(batch.keys()))
    break
"""

if __name__ == "__main__":
    calculators = [GraphCalculator(), DescriptorCalculator()]
    builder = DatasetBuilder(raw_data_path="data\\raw\\chembl_compounds.csv", calculators=calculators)
    builder.build(sp="550_1000.csv")