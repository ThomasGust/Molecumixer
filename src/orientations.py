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


def get_orientation_vector(n, max_distance):
    base_vector = list(range(n))
    permuted_vector = base_vector.copy()
    
    max_distance = min(max_distance, n)

    for _ in range(max_distance):
        idx1, idx2 = random.sample(range(n), 2)

        permuted_vector[idx1], permuted_vector[idx2] = permuted_vector[idx2], permuted_vector[idx1]
        current_distance = sum([1 for i, j in zip(base_vector, permuted_vector) if i != j])

        if current_distance >= max_distance:
            break
    
    return permuted_vector

def shuffle_n_m_matrix(matrix, new_orientation):
    # In the future, we might want this function to randomize where it places the odd numbered chunk


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

def permute_nodes(graph, chunks, maximum_hamming_distance):
    orientation_vector = get_orientation_vector(chunks, maximum_hamming_distance)
    node_matrix = graph.x
    permuted_matrix = shuffle_n_m_matrix(node_matrix, orientation_vector)
    x = {"x":torch.tensor(permuted_matrix), "orientation":orientation_vector}

    return x

def permute_edges(graph, chunks, maximum_hamming_distance):
    # Permutes on side of the edge index and the whole edge attributes according to a permutation vector with a fixed hamming distance
    
    orientation_vector = get_orientation_vector(chunks, maximum_hamming_distance)
    print(orientation_vector)
    edge_index = torch.permute(graph.edge_index, (1,0))
    
    top = edge_index[:, 0][:, None]

    permuted_top = shuffle_n_m_matrix(top, orientation_vector)

    new_edge_index = torch.zeros_like(edge_index)
    new_edge_index[:, 0] = torch.tensor(permuted_top[:, 0])
    new_edge_index[:, 1] = edge_index[:, 1]
    
    edge_attributes = graph.edge_attr
    permuted_edge_attributes = shuffle_n_m_matrix(edge_attributes, orientation_vector)

    x = {"edge_index":torch.permute(new_edge_index, (1, 0)), "edge_attr": torch.tensor(permuted_edge_attributes), "orientation":orientation_vector}
    
    return x

def split_tensor(tensor, batch_index):
    batch_index = batch_index.to(torch.int64)
    unique_batches = torch.unique(batch_index)
    split_tensors = []

    for batch_id in unique_batches:
        mask = (batch_index == batch_id)
        split_tensors.append(tensor[mask])

    return split_tensors

def permute_each_nodes(graphs, chunks, maximum_hamming_distance):
    split_nodes = split_tensor(graphs.x, graphs.batch)