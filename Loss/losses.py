import torch
import numpy as np

def vorticity_residual(w, re=1000.0):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, :], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1, 1, N, N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1, 1, N, N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy
    diffusion = (1.0 / re) * wlap

    return advection, diffusion

def voriticity_residual_three_channel(w, re=1000.0, dt=1/32):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)
    diffusion = (1.0 / re) * wlap
    residual = wt + (advection - diffusion + 0.1*w[:, 1:-1]) - f
    residual_loss = ((residual ** 2).mean())
    return residual_loss


def boundary_condition_residual(w):
    w = w.clone()
    w.requires_grad_(True)
    left_edge = w[:, :, :, 0]
    right_edge = w[:, :, :, -1]
    top_edge = w[:, :, 0, :]
    bottom_edge = w[:, :, -1, :]
    return left_edge, right_edge, top_edge, bottom_edge

def l2_loss(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()

def calculate_loss(config, model, xt, data, t):
    '''
    This function is for calculate the combined loss(RMSE + PINNS) from the model output
    Input:
        config file
        model : u-net
        xt : the corrupted version of x0
        data : HR reference
        t : timestep t
    Output:
        loss : combined loss
    '''

    output = model(xt, t)
    # Regular MSE
    loss_mse = l2_loss(output, data)
    # Advection and Diffusion Loss
    adv, dif = vorticity_residual(data)
    adv_pred, dif_pred = vorticity_residual(output)
    loss_adv = l2_loss(adv, adv_pred)
    loss_dif = l2_loss(dif, dif_pred)
    # BC Loss
    left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(output)
    loss_bc = l2_loss(left_edge, right_edge) + l2_loss(topt_edge, bottom_edge)
    loss = (config.train.MSE_weight * loss_mse
            + config.train.ADV_weight * loss_adv
            + config.train.DIF_weight * loss_dif
            + config.train.BC_weight * loss_bc)
    return loss, loss_mse, loss_adv, loss_dif, loss_bc

def calculate_loss_dev(config, output, data):
    '''
    This function is for calculate the combined loss(RMSE + PINNS) from the model output
    Input:
        config file
        model : u-net
        xt : the corrupted version of x0
        data : HR reference
        t : timestep t
    Output:
        loss : combined loss
    '''
    # Regular MSE
    loss_mse = l2_loss(output, data)
    # Advection and Diffusion Loss
    adv, dif = vorticity_residual(data)
    adv_pred, dif_pred = vorticity_residual(output)
    loss_adv = l2_loss(adv, adv_pred)
    loss_dif = l2_loss(dif, dif_pred)
    # BC Loss
    left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(output)
    loss_bc = l2_loss(left_edge, right_edge) + l2_loss(right_edge, left_edge)
    loss = (config.train.MSE_weight * loss_mse
            + config.train.ADV_weight * loss_adv
            + config.train.DIF_weight * loss_dif
            + config.train.BC_weight * loss_bc)
    return loss, loss_mse, loss_adv, loss_dif, loss_bc

def calculate_loss_three_channel(config, model, xt, data, t, scaler):
    '''
    This function is for calculate the combined loss(RMSE + PINNS) from the model output
    Input:
        config file
        model : u-net
        xt : the corrupted version of x0
        data : HR reference
        t : timestep t
    Output:
        loss : combined loss
    '''
    output = model(xt, t)
    # Regular MSE
    loss_mse = l2_loss(output, data)
    # Advection and Diffusion Loss
    residual = voriticity_residual_three_channel(scaler.inverse(output))
    # BC Loss
    left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(output)
    loss_bc = l2_loss(left_edge, right_edge) + l2_loss(top_edge, bottom_edge)
    loss = 1 * loss_mse + 0 * residual/1000 + 0 * loss_bc
    return loss, loss_mse, residual, loss_bc

def calculate_loss_dev_three_channel(config, output, data, scaler):
    '''
    This function is for calculate the combined loss(RMSE + PINNS) from the model output
    Input:
        config file
        model : u-net
        xt : the corrupted version of x0
        data : HR reference
        t : timestep t
    Output:
        loss : combined loss
    '''
    # Regular MSE
    loss_mse = l2_loss(scaler.inverse(output), scaler.inverse(data))
    residual = voriticity_residual_three_channel(scaler.inverse(output))
    # BC Loss
    left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(scaler.inverse(output))
    loss_bc = l2_loss(left_edge, right_edge) + l2_loss(right_edge, left_edge)
    loss = (1 * loss_mse
            + 0 * residual/1000
            + 0 * loss_bc)
    return loss, loss_mse, residual, loss_bc
