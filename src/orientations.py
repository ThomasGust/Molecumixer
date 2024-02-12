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
    graph.x = permuted_matrix

    return graph, orientation_vector

def permute_edges(graph, chunks, maximum_hamming_distance):
    # Permutes on side of the edge index and the whole edge attributes according to a permutation vector with a fixed hamming distance
    orientation_vector = get_orientation_vector(chunks, maximum_hamming_distance)
    edge_index = torch.permute(graph.edge_index, (1,0))
    top = edge_index[:, 0]
    permuted_top = shuffle_n_m_matrix(top, orientation_vector)

    edge_index[:, 0] = permuted_top
    permuted_edge_index = torch.permute(edge_index, (1,0))
    graph.edge_index = permuted_edge_index

    
    edge_attributes = graph.edge_attr
    permuted_edge_attributes = shuffle_n_m_matrix(edge_attributes, orientation_vector)
    graph.edge_attr = permuted_edge_attributes

    return graph, orientation_vector