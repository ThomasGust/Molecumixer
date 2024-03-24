from tasks import Task
import faiss
import pickle as pkl

def triplet_loss():
    pass

def euclidean_distance():
    pass

class Clusterer:
    """A KNN model from dense vectors. Early in training, these will be fixed clusters based off a dense molecular fingerprint and chemical descriptors.
       Later in training, after the model has some understanding of chemical space, we will introduce a second clustering task based off of KNN neighborhoods
       of our models latent representation."""
    
    def __init__(self, d, ncentroids, niter, verbose=True):
        self.d = d
        self.ncentroids = ncentroids
        self.niter = niter
        self.verbose = verbose

        self.kmeans = faiss.Kmeans(d, ncentroids, niter=self.niter, verbose=self.verbose, gpu=True)
    
    def train(self, x):
        self.kmeans.train(x)
    
    def save(self, sp):
        with open(sp, 'wb') as f:
            pkl.dump(self.kmeans.centroids, f)
    
    def load(self, sp):
        with open(sp, "rb") as f:
            centroids = pkl.load(f)
            self.kmeans.centroids = centroids
    
    def search(self, x, c=1):
        d, i = self.kmeans.index.search(x, c)
        return d, i

class ClusterPredictionModel:
    """Given a latent vector from an encoder, this module will predict pseudolabel clusters"""

    def __init__(self):
        pass

class ClusterPredictionTask(Task):
    """Implements the cluster prediction training task"""
    
    def __init__(self):
        super().__init__("cluster_prediction")

class ClusterContrastiveLearningTask(Task):
    """Implements a cluster contrastive learning pretraining task"""

    def __init__(self):
        super().__init__("cluster_contrastive_learning")