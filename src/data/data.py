from utils import pathjoin
import os
import pickle as pkl
from utils import from_smiles
from tasks import DescriptorCalculator, FingerprintCalculator
from tqdm import tqdm
import pandas as pd
from rdkit.Chem import MolFromSmiles
from functools import reduce

class GraphCalculator:
    """Builds the root of our dataset by computing molecular graphs and basic metadata from smiles and ChEMBL ID"""

    def __init__(self):
        pass

    def calculate(self, raw_data):
        smiles = raw_data['Smiles']
        ids = raw_data['ChEMBL ID']

        smiles = list(smiles)
        ids = list(ids)

        dataf  = {"x":[], "edge_index":[], "edge_attr":[], "smiles":[], "id":[]}

        i=0

        for mol in tqdm(smiles, desc='Building Graphs'):
            try:
                graph = from_smiles(mol, with_hydrogen=True, kekulize=False)
            except Exception as e:
                i += 1
                continue

            dataf['x'].append(graph['x'])
            dataf['edge_index'].append(graph['edge_index'])
            dataf['edge_attr'].append(graph['edge_attr'])
            dataf['smiles'].append(graph['smiles'])
            dataf['id'].append(ids[i])
            
            i += 1
    
        return dataf

class DatasetBuilder:

    def __init__(self, raw_data_path, calculators=None):

        self.raw_data_path = raw_data_path
        self.calculators = calculators

        self.raw_data = pd.read_csv(self.raw_data_path, sep=";") #TODO LOAD RAW DATA HERE
        self.raw_data = self.raw_data.iloc[550:650]
        self.keys = None
        print("DATASET BUILDER: LOADED RAW DATA")

    def build(self, sp=None):
        dfs = []
        for calculator in self.calculators:
            ret_d = calculator.calculate(self.raw_data)
            df = pd.DataFrame(ret_d)
            dfs.append(df)
        
        dataset = reduce(lambda df1, df2: pd.merge(df1, df2, on='id'), dfs)
        dataset.to_csv(sp)
        
    def __getitem__(self, idx):
        element = {key:self.dataset[key][idx] for key in self.keys}
        return element
        

class MolecumixerDataset:

    def __init__(self, build_directory):
        self.build_directory = build_directory
        
        self.dm = DataManager(self.build_directory)
        self.dm.merge()

        self.df = pkl.load(open(pathjoin(self.build_directory, "merge.pkl"), "rb"))
    
    def __len__(self):
        return len(list(self.df.keys()))
    
    def __getitem__(self, idx):
        resource = self.df[list(self.df.keys())[idx]]
        return resource

class DataManager:
    """This object will handle merging the task specific training files and processing them into dataloaders"""

    def __init__(self, build_directory):
        self.build_directory = build_directory

        ls = os.listdir(self.build_directory)
        try:
            ls.remove('merged.pkl')
        except ValueError:
            pass
    
        self.names = [d.split('.pkl')[0] for d in ls]
        self.build_paths = [pathjoin(self.build_directory, d) for d in ls]
        
        self.build_resources = {}
        print(self.names)
        for i, name in enumerate(self.names):
            data = pkl.load(open(self.build_paths[i], "rb"))
            self.build_resources[name] = data
        
    def merge(self):
        df = {}
        ids = []
        for ri in list(self.build_resources.keys()):
            resource = self.build_resources[ri]
            for id in list(resource.keys()):
                if id not in ids:
                    ids.append(id)
                    df[id] = {}
        
        for ri in list(self.build_resources.keys()):
            resource = self.build_resources[ri]
            for id in list(resource.keys()):
                data = resource[id]
                df[id][ri] = data
        
        with open(pathjoin(self.build_directory, "merge.pkl"), "wb") as f:
            pkl.dump(df, f)

    def create_dataloader(self):
        with open(pathjoin(self.build_directory, "merge.pkl"), "rb") as f:
            merged = pkl.load(f)
        
        print(list(merged.keys()))

if __name__ == "__main__":
    #manager = DataManager("data\\build", "data\\dataloader")
    calculators = [GraphCalculator(), DescriptorCalculator()]
    builder = DatasetBuilder(raw_data_path="data\\raw\\chembl_compounds.csv", calculators=calculators)