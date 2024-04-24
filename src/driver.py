#THIS IS A TEST FILE DUE TO THE WAY RELATIVE PYTHON PATHING WORKS
from torch.utils.data import DataLoader
from data.data import DataManager, MolecumixerDataset, DatasetBuilder, DescriptorCalculator, GraphCalculator, FingerprintCalculator, ClusterCalculator
import time
import torch

if __name__ == "__main__":
    calculators = [GraphCalculator(), FingerprintCalculator(), DescriptorCalculator(include_g3=True), ClusterCalculator(396, 10, 20)]
    builder = DatasetBuilder(raw_data_path="data\\raw\\chembl_compounds.csv", calculators=calculators)
    builder.build(sp="550_1000.csv")

    #WORK HARDER NOT SMARTER