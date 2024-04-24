#from tasks import Task
import faiss
import pickle as pkl
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def triplet_loss():
    pass

def euclidean_distance():
    pass
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

class ClusterCalculator:
    """
    A KNN model from dense vectors. Early in training, these will be fixed clusters based off a dense molecular fingerprint and chemical descriptors.
       Later in training, after the model has some understanding of chemical space, we will introduce a second clustering task based off of KNN neighborhoods
       of our models latent representation.
       """
    
    def __init__(self, d, ncentroids, niter, verbose=True):
        self.d = d
        self.ncentroids = ncentroids
        self.niter = niter
        self.verbose = verbose

        #self.kmeans = faiss.Kmeans(d, ncentroids, niter=self.niter, verbose=self.verbose, gpu=True)
    
    def train(self, x):
        self.kmeans.fit(x)
    
    def save(self, sp):
        with open(sp, 'wb') as f:
            pkl.dump(self.kmeans.cluster_centers_, f)
    
    def load(self, sp):
        with open(sp, "rb") as f:
            centroids = pkl.load(f)
            self.neighborhood = NearestNeighbors()
            self.neighborhood.fit(centroids)
    
    def search(self, x, c=1):
        neighbors = self.neighborhood.kneighbors(x, c)
        return neighbors
    
    def calculate(self, data):
        smiles = data['smiles']
        ids = data['id']
        descriptors = data['descriptors']
        struct_fp = data['maccs']

        descriptors = list(descriptors)
        struct_fp = list(struct_fp)
        smiles = list(smiles)
        ids = list(ids)

        retids = []

        clusters = []
        train_data = []

        cntr = 0
        for descriptor in tqdm(descriptors, desc=f"Generating Training Data for Cluster Calculator With Coarseness {self.ncentroids}"):
            #print(descriptor.shape)
            #print(struct_fp[cntr].shape)
            d = torch.cat((descriptor, struct_fp[cntr]))
            train_data.append(d)
            cntr += 1
        

        train_data = np.array(train_data, dtype=np.float64)
        train_data = np.nan_to_num(train_data, nan=0.0)
        train_data = pd.DataFrame(train_data)
        train_data = clean_dataset(train_data)
        train_data = torch.tensor(np.array(train_data, dtype=np.float64), dtype=torch.float64)
        train_data = train_data[~torch.any(train_data.isnan(),dim=1)]

        """
        nans = torch.isnan(train_data)
        print(nans)
        print(torch.unique(nans))
        infs = torch.isinf(train_data)
        print(infs)
        print(torch.unique(infs))
        """
        #print(train_data)

        print(f"Starting to Train Clustering Model With Coarseness: {self.ncentroids}")
        self.kmeans = KMeans(self.ncentroids)

        self.train(train_data)
        self.save("cluster_centroids.pkl")
        self.load("cluster_centroids.pkl")

        cntr = 0
        for element in tqdm(train_data, f"Calculating Clusters with Coarseness {self.ncentroids}"):
            element_clusters = self.search(element.reshape(1, -1), c=5)
            retids.append(ids[cntr])
            clusters.append(element_clusters)
            cntr += 1
        
        return {"clusters":clusters, "id":retids}

class ClusterPredictionModel(nn.Module):
    """Given a latent vector from an encoder, this module will predict pseudolabel clusters"""

    def __init__(self):
        super(ClusterPredictionModel, self).__init__()

if __name__ == "__main__":
    data = np.random.randn(1_000_000, 500)*np.random.randn(1_000_000, 500)
    print(data.shape)
    calculator = ClusterCalculator(500, 100, 20)
    calculator.train(data)
