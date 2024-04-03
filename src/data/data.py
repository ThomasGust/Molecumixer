from utils import pathjoin
import os
import pickle as pkl

class DataManager:
    """This object will handle merging the task specific training files and processing them into dataloaders"""

    def __init__(self, build_directory, dataloader_directory):
        self.build_directory = build_directory
        self.dataloader_directory = dataloader_directory

        ls = os.listdir(self.build_directory)
        try:
            ls.remove('merged.pkl')
        except ValueError:
            pass
    
        self.names = [d.split('.pkl')[0] for d in ls]
        self.build_paths = [pathjoin(self.build_directory, d) for d in ls]
        
        self.build_resources = {}
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
        print(merged)
if __name__ == "__main__":
    manager = DataManager("data\\build", "data\\dataloader")
