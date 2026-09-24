import os
import yaml

import hydra
import lightning as L
import torch
from generative.networks.nets import DiffusionModelUNet
from generative.inferers import DiffusionInferer
from omegaconf import OmegaConf
from monai.utils import first
import torch.nn as nn
import torch.nn.functional as F


class DDPM(L.LightningModule):
    def __init__(self, autoencoder, params, scheduler, optimizer):
        super().__init__()
        
        dict_params = \
            {key: value
             for param in OmegaConf.to_container(params, resolve=True)
             for key, value in param.items()}

        self.autoencoder = self.load_autoencoder(autoencoder)
        self.unet = DiffusionModelUNet(**dict_params)
        self.scheduler = scheduler
        self.inferer = None
        self.scale_factor = None

        self.inferer = DiffusionInferer(scheduler=self.scheduler)
        self.optimizer = optimizer(self.unet.parameters())

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=['scheduler'])

    def forward(self, x):
       x_recon, _, _, _ = self.autoencoder(x)
       print(x_recon.shape)

       self.scheduler.set_timesteps(num_inference_steps=100)
       noise = torch.randn_like(x)
       image = self.inferer.sample(input_noise=noise,
                                   diffusion_model=self.unet,
                                   scheduler=self.scheduler,
                                   conditioning=x_recon,
                                   mode='concat')
       return image

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]

        x_recon, _, _, _ = self.autoencoder(x)

        noise = torch.randn_like(x_recon)

        timesteps = torch.randint(
                0, self.inferer.scheduler.num_train_timesteps, (x.shape[0],), device=self.device
            ).long()
            
        noise_pred = self.inferer(inputs=x,
                                  diffusion_model=self.unet,
                                  noise=noise,
                                  timesteps=timesteps,
                                  condition=x_recon,
                                  mode='concat')
        
        loss = F.mse_loss(noise_pred.float(), noise.float())
        self.log('train_loss', loss, batch_size=batch_size,
                  on_step=False, on_epoch=True, prog_bar=True, logger=True,
                  sync_dist=True)

        return loss

    def load_autoencoder(self, autoencoder):
        path_weights = autoencoder.weights
        path_config = autoencoder.config

        assert os.path.exists(path_weights), f'Weights not found in {path_weights}'
        assert os.path.exists(path_config), f'Config not found in {path_config}'

        with open(path_config) as stream:
            config = yaml.safe_load(stream)

        config['model']
        model = hydra.utils.instantiate(config['model'])
        model.load_from_checkpoint(path_weights)

        for param in model.parameters():
            param.requires_grad = False

        autoencoder = model.model

        return autoencoder
