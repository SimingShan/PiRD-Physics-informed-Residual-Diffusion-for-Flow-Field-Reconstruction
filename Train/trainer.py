from Diffusion.diffusion import diffusion
from Loss.losses import calculate_loss, calculate_loss_three_channel
from tqdm import tqdm
import torch
from utils.datasets import corrupt_and_upscale_image
from Loss.losses import l2_loss, voriticity_residual_three_channel, boundary_condition_residual
torch.set_printoptions(edgeitems=3, linewidth=200, threshold=5000)
def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training Epoch')
    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        x_t, t = diffusion(config, data, device)
        loss, loss_mse, loss_adv, loss_dif, loss_bc = calculate_loss(config, model, x_t, data, t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_description(f'Loss: {loss.item():.4f}'
                                     f'Loss_MSE: {loss_mse.item():.4f}'
                                     f'Loss_DEV: {loss_adv.item():.4f}'
                                     f'Loss_DIF: {loss_dif.item():.4f}'
                                     f'Loss_BC: {loss_bc.item():.4f}')
    return total_loss / len(dataloader)

def train_epoch_three_channel(model, dataloader, optimizer, config, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training Epoch')
    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        x_t, t = diffusion(config, data, device)
        loss, loss_mse, residual, loss_bc = calculate_loss_three_channel(config, model, x_t, data, t, scaler)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_description(f'Loss: {loss.item():.4f}'
                                     f'Loss_L2: {loss_mse.item():.4f}'
                                     f'Loss_residual: {residual.item():.4f}'
                                     f'Loss_BC: {loss_bc.item():.4f}')
    return total_loss / len(dataloader)


def train_unet(model, dataloader, optimizer, config, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training Epoch')
    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        y0 = corrupt_and_upscale_image(config, data)
        output = model(y0)
        # Regular MSE
        loss_mse = l2_loss(output, data)
        residual = voriticity_residual_three_channel(scaler.inverse(output))
        # BC Loss
        left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(output)
        loss_bc = l2_loss(left_edge, right_edge) + l2_loss(right_edge, left_edge)
        loss = (0.7 * loss_mse + 0.2 * residual/1000 + 0.005 * loss_bc)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_description(f'Loss: {loss.item():.4f}'
                                     f'Loss_L2: {loss_mse.item():.4f}'
                                     f'Loss_residual: {residual.item():.4f}'
                                     f'Loss_BC: {loss_bc.item():.4f}')
    return total_loss / len(dataloader)

