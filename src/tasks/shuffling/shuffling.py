import numpy as np
import torch
import math
import numpy as np
import random

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

    def __init__(self, chunks, maximum_hamming_distance):
        self.chunks = chunks
        self.maximum_hamming_distance = maximum_hamming_distance
    
    def get_orientation_vector(self):
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

    def permute_nodes(self, graph, chunks, maximum_hamming_distance):
        orientation_vector = self.get_orientation_vector(chunks, maximum_hamming_distance)
        node_matrix = graph.x
        permuted_matrix = self.shuffle_node_matrix(node_matrix, orientation_vector)
        x = {"x":torch.tensor(permuted_matrix), "orientation":orientation_vector}

        return x


    def shuffle_node_matrix(self, matrix, new_orientation):
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

    def permute_nodes(self, graph, chunks, maximum_hamming_distance):
        orientation_vector = self.get_orientation_vector(chunks, maximum_hamming_distance)
        node_matrix = graph.x
        permuted_matrix = self.shuffle_node_matrix(node_matrix, orientation_vector)
        x = {"x":torch.tensor(permuted_matrix), "orientation":orientation_vector}

        return x


class ShufflingModel:
    """Given an encoder, this model predicts how a graph was shuffled/permuted"""
