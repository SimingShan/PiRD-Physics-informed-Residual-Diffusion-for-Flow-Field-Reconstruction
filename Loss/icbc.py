import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = np.load('train.npy')
data = torch.from_numpy(data)
def boundary_condition_residual(w):
    w = w.clone()
    w.requires_grad_(True)
    left_edge = w[:, :, :, 0]
    right_edge = w[:, :, :, -1]
    top_edge = w[:, :, 0, :]
    bottom_edge = w[:, :, -1, :]
    return left_edge, right_edge, top_edge, bottom_edge

left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(data[:2, :1, ...])
print(((left_edge-right_edge)**2).mean())
print(((top_edge-bottom_edge)**2).mean())

def enforce_periodicity(data):
    """
    Enforce periodicity on the downsampled data by making the boundaries (edges) of the data
    more similar to their opposite edges through averaging.

    Parameters:
    - data: 4D numpy array with shape [num_sequences, time_steps, height, width]

    Returns:
    - Modified data with enforced periodicity.
    """
    # Copy the data to avoid modifying the original in-place
    modified_data = data.copy()

    # Loop through all sequences and time steps
    for seq in range(data.shape[0]):
        for time in range(data.shape[1]):
            # Extract edges
            left_edge = data[seq, time, :, 0]
            right_edge = data[seq, time, :, -1]
            top_edge = data[seq, time, 0, :]
            bottom_edge = data[seq, time, -1, :]

            # Average edges with their opposite edges
            modified_data[seq, time, :, 0] = (left_edge + right_edge) / 2
            modified_data[seq, time, :, -1] = (left_edge + right_edge) / 2
            modified_data[seq, time, 0, :] = (top_edge + bottom_edge) / 2
            modified_data[seq, time, -1, :] = (top_edge + bottom_edge) / 2

            # Optionally, apply similar averaging for corner points if needed

    return modified_data
