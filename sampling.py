import torch
from datasets import FlowDataset, corrupt_and_upscale_image
from diffusion import extract_into_tensor, get_schedule
import yaml
import numpy as np
from config_util import AppConfig
from datasets import show_blur_image, StdScaler
from losses import vorticity_residual
from u_net_attention import ConditionalModel
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

with open('config.yml') as f:
    raw_config = yaml.safe_load(f)
config = AppConfig(**raw_config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConditionalModel(config).to(device)
mode = 'physics'
if mode == 'baseline':
    save_path = "output_model/base_model_params_epoch_60_0.0002_30_0.5_skip_20step.pth"
    model.load_state_dict(torch.load(save_path, map_location=device))
elif mode == 'physics':
    save_path = "output_model/model_params_epoch_80_0.0002_30_0.5_skip_20step.pth"
    model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
sqrt_beta = get_schedule(config)
beta = sqrt_beta ** 2
beta_prev = np.append(0.0, beta[:-1])
alpha = beta - beta_prev
posterior_variance = config.diffusion.kappa**2 * beta_prev / beta * alpha
posterior_variance_clipped = np.append(
                posterior_variance[1], posterior_variance[1:]
                )
posterior_log_variance_clipped = np.log(posterior_variance_clipped)
posterior_mean_coef1 = beta_prev / beta
posterior_mean_coef2 = alpha / beta
dataset = FlowDataset(path='train.npy', transform=config.dataset.transform, process='test')
scaler = dataset.transform
dataset = dataset[420, ...].unsqueeze(0).to(device) #420
show_blur_image(dataset, 0, 0, 'HR reference')
y0 = corrupt_and_upscale_image(config, data=dataset).to(device)
show_blur_image(y0, 0, 0, 'y0 LR image')
noise = torch.randn(1, 1, 256, 256).to(device)
x_prev = (y0 + config.diffusion.kappa * noise).to(device)
show_blur_image(x_prev, 0, 0, 'x_T noised HR')
for i in list(range(config.diffusion.steps))[::-1]:
    x_prev = x_prev.to(device)
    var = extract_into_tensor(posterior_variance_clipped, i, broadcast_shape=[1, 1, 256, 256])
    sd = np.sqrt(var).to(device)
    log_variance = extract_into_tensor(posterior_log_variance_clipped, i, broadcast_shape=[1, 1, 256, 256]).to(device)
    coef1 = extract_into_tensor(posterior_mean_coef1, i, broadcast_shape=[1, 1, 256, 256]).to(device)
    coef2 = extract_into_tensor(posterior_mean_coef2, i, broadcast_shape=[1, 1, 256, 256]).to(device)
    t = (torch.Tensor(1) + i).to(device)

    if i != 0:
        x_prev = coef1 * x_prev + coef2 * model(x_prev, t) + torch.exp(0.5*log_variance) * torch.randn(1, 1, 256, 256).to(device)

    else:
        x_0 = model(x_prev, t)
        x_0 = scaler.inverse(x_0)
        output = x_0.cpu().detach()
        show_blur_image(output, 0, 0, f'x_{i}')

from PIL import Image
path1 = "output_image/HR reference.png"
path2 = "output_image/y0 LR image.png"
path3 = "output_image/x_0.png"
# Replace these with your actual file paths
image_paths = [path1, path2, path3]
images = [Image.open(image) for image in image_paths]

# Assuming all images have the same height and width
total_width = sum(image.width for image in images)
max_height = max(image.height for image in images)  # Since they are the same, you could also directly use one image's height

# Create a new image with the calculated total width and max height
new_image = Image.new('RGB', (total_width, max_height))

# Paste the images next to each other
x_offset = 0
for image in images:
    new_image.paste(image, (x_offset, 0))
    x_offset += image.width

# Save the new image
new_image.save(f'combined_image_{mode}.jpg')