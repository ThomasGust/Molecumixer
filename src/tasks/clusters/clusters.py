import faiss

class Clusterer:
    """A KNN model from dense vectors. Early in training, these will be fixed clusters based off a dense molecular fingerprint and chemical descriptors.
       Later in training, after the model has some understanding of chemical space, we will introduce a second clustering task based off of KNN neighborhoods
       of our models latent representation."""

class ClusterPredictionModel:
    """Given a latent vector from an encoder, this module will predict pseudolabel clusters"""