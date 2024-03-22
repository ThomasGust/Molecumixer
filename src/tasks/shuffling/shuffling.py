from tasks import Task

import numpy as np
import torch
import math
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

def compute_hamming_distance(v):
    """This function computes the hamming distance between a permutation vector and the base permutation
        It counts the spots that don't equal one another
    """
    c = len(v)
    base = [i for i in range(c)]
    return sum([0 if v1 == v2 else 1 for v1, v2 in list(zip(base, v))])

def split_tensor(t, batch_index):
    """
    This function splits a tensor from a batch index.
    if t is [0, pi, e, 6] for example, and the batch index is [0, 1, 0, 1],
    this function should return two tensors: [0, e] and [pi, 6]
    """
    batch_index = batch_index.to(torch.int64)
    unique_batches = torch.unique(batch_index)
    split_tensors = []

    for batch_id in unique_batches:
        mask = (batch_index == batch_id)
        split_tensors.append(t[mask])

    return split_tensors

class NodeShuffler:
    """
    This object handles all of the logic for shuffling nodes.
    Using a NodeShuffler, if we have a node matrix with a shape (NUM_ATOMS, NUM_ATTRIBUTES)
    we can reshuffle it into a matrix of the same shape where chunks of the matrix have been moved around a little bit.
    We generate a vector, something like [1, 0, 3, 2] a given max hamming distance away from [0, 1, 2, 3] and use that to shuffle the matrix.
    As an example, with the above shuffling vector [0, 1, 2, 3, 4, 5, 6, 7] would become [2, 3, 0, 1, 6, 7, 3, 4]. The same idea is applied to our graphs,
    except we take into account another dimension to handle the different node level attributes.
    """
    def __init__(self, chunks, maximum_hamming_distance):
        self.chunks = chunks
        self.maximum_hamming_distance = maximum_hamming_distance
    
    def get_orientation_vector(self):
        """
        Computes a pseudorandom shuffling vector using the number of chunks and maximum hamming distance defined in our constructor.
        """
        base_vector = list(range(self.chunks))
        permuted_vector = base_vector.copy()
        
        max_distance = min(self.maximum_hamming_distance, self.chunks)

        for _ in range(max_distance):
            idx1, idx2 = random.sample(range(self.chunks), 2)

            permuted_vector[idx1], permuted_vector[idx2] = permuted_vector[idx2], permuted_vector[idx1]
            current_distance = sum([1 for i, j in zip(base_vector, permuted_vector) if i != j])

            if current_distance >= max_distance:
                break
                
        return permuted_vector

    def permute_nodes(self, graph):
        """
        Main function for this object, given a graph, it will generate an orientation vector,
        shuffle the original graphs node matrix, and return both our new node matrix, and the shuffle vector used to generate it.
        """
        orientation_vector = self.get_orientation_vector(self.chunks, self.maximum_hamming_distance)
        node_matrix = graph.x
        permuted_matrix = self.shuffle_node_matrix(node_matrix, orientation_vector)
        x = {"x":torch.tensor(permuted_matrix), "orientation":orientation_vector}

        return x


    def shuffle_node_matrix(self, matrix, new_orientation):
        """
        This function handles the actual shuffling of a node matrix based off of an orientation vector.
        """
        n, m = matrix.shape
        num_chunks = len(new_orientation)
        
        permuted_matrix = np.zeros_like(matrix)
        chunk_height = int(math.floor(n/num_chunks))
        
        for new_pos, original_pos in enumerate(new_orientation):
            
            original_row_start = original_pos * chunk_height
            new_row_start = new_pos * chunk_height
            current_chunk_height = chunk_height if (original_pos < num_chunks - 1) else n - original_row_start
            
            permuted_matrix[new_row_start:new_row_start+current_chunk_height, :] = matrix[original_row_start:original_row_start+current_chunk_height, :]

        return permuted_matrix


class ShufflingModel(nn.Module):
    """Given an encoder, this model predicts how a graph was shuffled/permuted"""

    def __init__(self, encoder_dim, hidden_dim, chunks, activation = F.relu):
        super().__init__()

        self.chunks = chunks
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.chunks^2

        self.activation = activation

        self.l1 = nn.Linear(self.encoder_dim, self.hidden_dim)
        self.o = nn.Linear(self.hidden_dim, self.output_dim) # This head generates the entire sequence at once
    
    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        
        x = self.o(x)

        x = torch.reshape(x, (self.output_dim, self.output_dim))
        return x # No activation is applied to the last layer

    def get_loss(self, labels, logits):
        """
        Computes the loss of our model based off of the output of forward [batch_size, chunks, chunks]
        and labels of shape [batch_size, chunks]
        """
        logits = logits.view(-1, self.chunks)
        labels = labels.view(-1)

        cel = nn.CrossEntropyLoss()
        loss = cel(logits, labels)

        return loss

class ShufflingPredictionTask(Task):
    """Implements the shuffling prediction pretraining task"""
    
    def __init__(self):
        super().__init__()