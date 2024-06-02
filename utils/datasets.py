import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from scipy.spatial import cKDTree

import numpy as np
import torch

def load_flow_data(path, process=None):
    '''
    Load flow data from path [N, T, h, w]
    Flatten the data into shape [N * T, 1, h, w]
    Return the flattened data, mean, and sd
    Load only the specified subset of the data based on the process parameter
    '''
    # Load data
    data = np.load(path)
    data_mean, data_scale = np.mean(data), np.std(data)
    print('Original data shape:', data.shape)

    # Split the data based on the process parameter
    if process == 'train':
        N = int(data.shape[0] * 0.8)  # Use 80% for training
        data = data[:N, ...]
    elif process == 'dev':
        N_start = int(data.shape[0] * 0.8)
        N_end = int(data.shape[0] * 0.9)  # Use next 10% for dev/validation
        data = data[N_start:N_end, ...]
    elif process == 'test':
        N_start = int(data.shape[0] * 0.9)  # Use last 10% for testing
        data = data[N_start:, ...]
    else:
        raise ValueError("please choose which dataset you are using (train, dev, or test)")
    print(f'Data range: mean: {data_mean}, scale: {data_scale}')

    # Convert data to torch.Tensor and flatten
    data = torch.tensor(data, dtype=torch.float32)  # Use torch.tensor() directly
    N, T, h, w = data.shape
    flattened_data = data.view(N * T, 1, h, w)  # Use view for efficient reshaping

    print(f'Flattened data shape: {flattened_data.shape}')
    return flattened_data, data_mean, data_scale

def load_flow_data_three_channel(path, process=None):
    # load flow data from path
    data = np.load(path)   # [N, T, h, w]

    print('Original data shape:', data.shape)
    data_mean, data_scale = np.mean(data), np.std(data)

    if process == 'train':
        N = int(data.shape[0] * 0.8)  # Use 80% for training
        data = data[:N, ...]
    elif process == 'dev':
        N_start = int(data.shape[0] * 0.8)
        N_end = int(data.shape[0] * 0.9)  # Use next 10% for dev/validation
        data = data[N_start:N_end, ...]
    elif process == 'test':
        N_start = int(data.shape[0] * 0.9)  # Use last 10% for testing
        data = data[N_start:, ...]
    else:
        raise ValueError("please choose which dataset you are using (train, dev, or test)")
    print(f'Data range: mean: {data_mean}, scale: {data_scale}')

    data = torch.as_tensor(data, dtype=torch.float32)
    flattened_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-2):
            flattened_data.append(data[i, j:j+3, ...])
    flattened_data = torch.stack(flattened_data, dim=0)
    print(f'data shape: {flattened_data.shape}')
    return flattened_data, data_mean, data_scale

class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std


def nearest(data):
    # Ensure arr is a numpy array
    arr = data.cpu()
    arr = np.array(arr)

    # Process each 2D slice individually
    for i in range(arr.shape[0]):  # Iterate over batch
        for j in range(arr.shape[1]):  # Iterate over channels
            # Extract the 2D slice
            slice_2d = arr[i, j, :, :]

            # Find the indices of zero and non-zero elements in the 2D slice
            zero_indices = np.argwhere(slice_2d == 0)
            non_zero_indices = np.argwhere(slice_2d != 0)

            # If there are no zero indices or non-zero indices, continue to the next slice
            if zero_indices.size == 0 or non_zero_indices.size == 0:
                continue

            # Build a KD-Tree from the non-zero indices for efficient nearest neighbor search
            tree = cKDTree(non_zero_indices)

            # For each zero index, find the nearest non-zero index and get its value
            for zero_idx in zero_indices:
                distance, nearest_idx = tree.query(zero_idx)
                nearest_non_zero_idx = non_zero_indices[nearest_idx]
                # Replace the zero value with the nearest non-zero value
                arr[i, j, zero_idx[0], zero_idx[1]] = slice_2d[tuple(nearest_non_zero_idx)]

    return arr

def upscale_image(method, data):
    if method == 'portion':
        data = nearest(data)
        data = torch.from_numpy(data).float()
        data = F.interpolate(data, size=(256, 256), mode='nearest')
    else:
        data = F.interpolate(data, size=(256, 256), mode='bicubic')
    return data

def corrupt_and_upscale_image(config, data):
    method = config.corruption.method
    data = data
    scale = config.corruption.scale
    portion = config.corruption.portion
    if method == 'skip':
        blur_data = data[:, :, ::scale, ::scale]

    elif method == 'average':
        blur_data = torch.nn.functional.avg_pool2d(data, kernel_size=scale, stride=scale, padding=0)

    elif method == 'portion':
        if portion is None:
            raise ValueError("Portion must be specified for the 'portion' method.")

        N, C, H, W = data.shape
        total_pixels = H * W
        pixels_to_keep = int(total_pixels * portion)

        # Create a random mask for the entire batch and all channels at once
        flat_indices = torch.randperm(total_pixels)[:pixels_to_keep]
        mask = torch.zeros((N, C, total_pixels), dtype=torch.bool, device=data.device)
        mask[:, :, flat_indices] = True
        mask = mask.view(N, C, H, W)

        # Apply the mask
        blur_data = data * mask.float()
        print(f"corrupted using {method}")
    upscaled_data = upscale_image(method, blur_data)
    return upscaled_data


def show_blur_image(data, num, chan, n):
    '''
    Display a blurred image and save the plot with minimal white space.

    Parameters:
    - data: A tensor representing the batch of images, with shape [N, C, H, W].
    - num: The index of the image in the batch to display.
    - chan: The channel of the image to display.
    - n: The timestep or identifier for the image being processed.
    '''
    if data.dim() != 4:
        raise ValueError("Expected data to have shape [N, C, H, W]")
    image_data = data[num, chan]
    fig, ax = plt.subplots()  # Use subplots to get more control over layout
    ax.imshow(image_data.cpu().numpy(), cmap='twilight')
    ax.axis('off')  # Turn off axis to remove ticks and labels

    plt.title(n, pad=20)  # Add a title with padding to ensure it's included

    # Convert n to a simple numeric or string value if it's a tensor
    # n_value = n.item() if torch.is_tensor(n) else n
    plt.savefig(f'output_image/{n}.png')
    # Save the plot with minimal white space
    #plt.show()



class FlowDataset(Dataset):
    '''
    Load the dataset
    Normalize the shape
    Get mean and sd
    Finally normalize it
    '''
    def __init__(self, path, process, transform=False):
        # Load data
        self.data, self.mean, self.sd = load_flow_data(path, process)
        # Set the transformation method based on the transform argument
        if transform == 'std':
            self.transform = StdScaler(self.mean, self.sd)  # Assuming StdScaler is a defined class
        elif transform is None:
            self.transform = None  # No transformation will be applied
        else:
            raise ValueError("Invalid normalization method specified. Choose 'std' or 'maxmin'.")

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the individual data sample
        data_sample = self.data[idx]

        # Apply transformations, if any
        if self.transform:
            data_sample = self.transform(data_sample)

        return data_sample

class FlowDataset_three_channel(Dataset):
    '''
    Load the dataset
    Normalize the shape
    Get mean and sd
    Finally normalize it
    '''
    def __init__(self, path, process, transform=False):
        # Load data
        self.data, self.mean, self.sd = load_flow_data_three_channel(path, process)
        # Set the transformation method based on the transform argument
        if transform == 'std':
            self.transform = StdScaler(self.mean, self.sd)  # Assuming StdScaler is a defined class
        elif transform is None:
            self.transform = None  # No transformation will be applied
        else:
            raise ValueError("Invalid normalization method specified. Choose 'std' or 'maxmin'.")

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the individual data sample
        data_sample = self.data[idx]

        # Apply transformations, if any
        if self.transform:
            data_sample = self.transform(data_sample)

        return data_sample


def create_dataloader(path, batch_size=32, shuffle=True, num_workers=0, transform=None):
    # Initialize the dataset
    dataset = FlowDataset(path=path, transform=transform)

    # Create and return the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader




