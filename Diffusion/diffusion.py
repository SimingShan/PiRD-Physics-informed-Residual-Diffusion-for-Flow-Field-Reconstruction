import math
import numpy as np
import torch
import random
from utils.datasets import corrupt_and_upscale_image

def get_schedule(config):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    schedule_name = config.diffusion.schedule_name
    num_diffusion_timesteps = config.diffusion.steps
    min_noise_level = config.diffusion.min_noise_level
    etas_end = 0.99
    kappa = config.diffusion.kappa
    kwargs = config.diffusion.kwargs
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

def extract_into_tensor(arr, timestep, broadcast_shape):
    """
    Extract a single value from a 1-D numpy array for all items in a batch based on a single timestep.

    :param arr: the 1-D numpy array.
    :param timestep: an integer index into the array to extract the single value.
    :param broadcast_shape: a shape to which the extracted value will be broadcast.
    :return: a tensor of the broadcast shape, filled with the value from the specified timestep.
    """
    # Ensure arr is a numpy array to support indexing directly
    arr = np.asarray(arr)

    # Convert the numpy array to a tensor and select the value at the given timestep
    selected_value = arr[timestep]

    # Create a tensor of zeros with the desired broadcast shape
    result_tensor = torch.zeros(broadcast_shape, dtype=torch.float32, device="cpu")  # Specify the device as needed

    # Broadcast the selected value across the entire tensor
    result_tensor += selected_value  # Correctly updates result_tensor

    return result_tensor

def prepare_diffusion_terms(arr, config, data, device):
    '''
    This function is aimed to return all necessary parameters needed in forward diffusion process
    Input:
        arr : the beta variance schedule
        data : the input data [N, C, W, H]
        config : the config file
    Output:
        t : a torch tensor of shape 1
        sqrt_n: extract sqrt(n_t) for the variance term in the forward diffusion process
        n: extract n_t for the mean term in the forward diffusion process
        noise: a random noise drawn from N(0,I) to be added to the variance
    '''
    sqrt_beta_list = arr
    beta_list = sqrt_beta_list ** 2
    t = random.randint(0, config.diffusion.steps - 1) #randomly draw a time t within the range of T
    sqrt_n = extract_into_tensor(sqrt_beta_list, t, data.shape).to(device)
    n = extract_into_tensor(beta_list, t, data.shape).to(device)
    noise = torch.randn_like(n).to(device)
    t = torch.Tensor([t]).to(device)
    assert sqrt_n.shape == n.shape == noise.shape == data.shape
    return t, sqrt_n, n, noise

def diffusion(config, data, device):
    '''
    :param arr:
    :param config:
    :param data:
    :param device:
    :return:
    '''
    sqrt_beta = get_schedule(config)
    y0 = corrupt_and_upscale_image(config, data)
    t, sqrt_n, n, noise = prepare_diffusion_terms(sqrt_beta, config, data, device)
    x_t = data + (y0 - data) * n + noise * config.diffusion.kappa * sqrt_n
    return x_t, t

