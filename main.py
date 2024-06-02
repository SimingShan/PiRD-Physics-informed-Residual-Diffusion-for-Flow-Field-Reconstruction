from torch.utils.data import DataLoader
import yaml
from utils.config_util import AppConfig
from Diffusion.u_net_attention import ConditionalModel
from utils.datasets import FlowDataset, FlowDataset_three_channel
from Train.trainer import train_epoch, train_epoch_three_channel, train_unet
import torch
from datetime import datetime
from Train.sampler import validation_three_channel, validation_unet


def load_config(path='configs/config.yml'):
    with open(path) as f:
        raw_config = yaml.safe_load(f)
    return AppConfig(**raw_config)

def setup_logger(log_file):
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    return log_message

method = 'three' #single or three
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()
    model = ConditionalModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epoch, eta_min=0.00001)
    # Initialize logging
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"log_file/log_{current_time}.txt"
    log = setup_logger(log_file)
    if method == 'single':
        dataset = FlowDataset(path=config.dataset.path, process='train', transform=config.dataset.transform)
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
        dataset_dev = FlowDataset(path=config.dataset.path, process='dev', transform=config.dataset.transform)
        dataloader_dev = DataLoader(dataset_dev, batch_size=config.train.batch_size, shuffle=True)
        log("Training with the method with SINGLE channel")
        for epoch in range(config.train.epoch):
            avg_loss = train_epoch(model, dataloader, optimizer, config, device)
            print(f'Epoch: {epoch}, Avg Training Loss: {avg_loss}')
            if (epoch + 1) % config.train.save_interval == 0:
                save_path = f"output_model/model_params_epoch_{epoch + 1}_three_channel.pth"
                checkpoint = {
                    'Method': "Describe your method here",
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'learning_rate': config.train.lr,
                    'loss_proportion': f'MSE:{config.train.MSE_weight},'
                                       f'ADV:{config.train.ADV_weight}'
                                       f'DIF:{config.train.DIF_weight}'
                                       f'BC:{config.train.BC_weight}',
                    'T': config.diffusion.steps
                }

                torch.save(checkpoint, save_path)
                print(f'Checkpoint saved to {save_path}')

            scheduler.step()
    elif method == 'three':
        dataset = FlowDataset_three_channel(path=config.dataset.path, process='train', transform=config.dataset.transform)
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
        scaler = dataset.transform
        dataset_dev = FlowDataset_three_channel(path=config.dataset.path, process='dev', transform=config.dataset.transform)
        dataloader_dev = DataLoader(dataset_dev, batch_size=config.train.batch_size, shuffle=True)
        log("Training with the method with THREE channel 20 steps kappa2 without pinns")
        best_dev_loss = float('inf')
        for epoch in range(config.train.epoch):
            avg_loss = train_epoch_three_channel(model, dataloader, optimizer, config, device, scaler)
            log(f'Epoch: {epoch}, Avg Training Loss: {avg_loss}')

            dev_loss = validation_three_channel(model, dataloader_dev, device, config, scaler)
            log(f'Epoch: {epoch}, Validation Set Loss: {dev_loss}')
 
            scheduler.step()
            if dev_loss < best_dev_loss:
                #log(f'Checkpoint saved for a validation loss of {dev_loss}')
                #best_dev_loss = dev_loss
                save_path = f"output_model/diffusion_kappa2_steps20_20epoch.pth"
                checkpoint = {
                    'Method': "Three channel,kappa=2, step=20, epoch=20, no pinns",
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'learning_rate': config.train.lr,
                    'loss_proportion': 'MSE:1',
                    'T': config.diffusion.steps,
                    'kappa': config.diffusion.kappa,
                    'scaler': config.corruption.scale
                }

                torch.save(checkpoint, save_path)
                print(f'Checkpoint saved to {save_path}')
                best_dev_loss = dev_loss
            #scheduler.step()
    elif method == 'unet':
        dataset = FlowDataset_three_channel(path=config.dataset.path, process='train', transform=config.dataset.transform)
        dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
        scaler = dataset.transform
        dataset_dev = FlowDataset_three_channel(path=config.dataset.path, process='dev', transform=config.dataset.transform)
        dataloader_dev = DataLoader(dataset_dev, batch_size=config.train.batch_size, shuffle=True)
        log("Training the simple unet with pinns")
        best_dev_loss = float('inf')
        for epoch in range(config.train.epoch):
            avg_loss = train_unet(model, dataloader, optimizer, config, device, scaler)
            log(f'Epoch: {epoch}, Avg Training Loss: {avg_loss}')

            dev_loss = validation_unet(model, dataloader_dev, device, config, scaler)
            log(f'Epoch: {epoch}, Validation Set Loss: {dev_loss}')
 
            scheduler.step()
            if dev_loss < best_dev_loss:
                log(f'Checkpoint saved for a validation loss of {dev_loss}')

                best_dev_loss = dev_loss
                save_path = f"output_model/unet_scale4_nearest.pth"
                checkpoint = {
                    'Method': "Three channel, simple unet, scale4, nearest",
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'learning_rate': config.train.lr,
                    'loss_proportion': 'MSE:1',
                    'T': config.diffusion.steps,
                    'kappa': config.diffusion.kappa,
                    'scaler': config.corruption.scale
                }

                torch.save(checkpoint, save_path)
                print(f'Checkpoint saved to {save_path}')
            scheduler.step()
            
                
    else:
        raise ValueError('Please specify with method you are training with')


if __name__ == '__main__':
    main()
