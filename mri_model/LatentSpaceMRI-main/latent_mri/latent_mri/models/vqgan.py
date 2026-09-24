import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from generative.networks.nets import VQVAE
from omegaconf import OmegaConf



class VQGAN(L.LightningModule):
    def __init__(self, weights, params, optimizer):
        super().__init__()
        dict_params_autoencoder = \
            {key: value
             for param in OmegaConf.to_container(params['autoencoder'], resolve=True)
             for key, value in param.items()}
        dict_params_discriminator = \
            {key: value
             for param in OmegaConf.to_container(params['discriminator'], resolve=True)
             for key, value in param.items()}
        
        self.model = VQVAE(**dict_params_autoencoder)
        self.discriminator = PatchDiscriminator(**dict_params_discriminator)

        self.optimizer_model = optimizer['autoencoder'](self.model.parameters())
        self.optimizer_discriminator = optimizer['discriminator'](self.discriminator.parameters())

        self.perceptual_loss = PerceptualLoss(spatial_dims=3,
                                              network_type='medicalnet_resnet10_23datasets',
                                              is_fake_3d=False)
        self.adv_loss = PatchAdversarialLoss(criterion='least_squares')

        self.adv_weight = weights['adv_weight']
        self.perceptual_weight = weights['perceptual_weight']
        self.recon_weight = weights['recon_weight']

        self.validation_step_dict = {}
        self.test_step_dict = {}
        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x):
        x_recon_logits, quant_loss = self.model(x)
        return x_recon_logits, quant_loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]

        optimizer_model, optimizer_discriminator = self.optimizers()

        x_recon_logits, quant_loss = self.model(x)
        recon_loss = self.recon_weight * F.l1_loss(x_recon_logits.float(), x.float())
        p_loss = self.perceptual_weight * self.perceptual_loss(x_recon_logits.float(), x.float())
        
        logits_fake = self.discriminator(x_recon_logits.contiguous().float())[-1]
        loss_g = self.adv_weight * self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        loss_model = recon_loss + p_loss + quant_loss + loss_g
 
        self.compute_optimization_step(self.model, optimizer_model, loss_model)

        logits_fake = self.discriminator(x_recon_logits.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        loss_d = 0.5 * self.adv_weight * (loss_d_fake + loss_d_real)
        self.compute_optimization_step(self.discriminator, optimizer_discriminator, loss_d)

        log_dict = {
            'train_loss': loss_model,
            'train_loss_gen': loss_g,
            'train_loss_disc': loss_d,
            'train_loss_recon': recon_loss,
            'train_loss_quant': quant_loss,
            'train_loss_percep': p_loss
        }

        self.log_dict(log_dict, batch_size=batch_size, on_step=False, 
                      on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        x_recon_logits, quant_loss = self.model(x)
        recon_loss = self.recon_weight * F.l1_loss(x_recon_logits.float(), x.float())
        p_loss = self.perceptual_weight * self.perceptual_loss(x_recon_logits.float(), x.float())
        
        logits_fake = self.discriminator(x_recon_logits.contiguous().float())[-1]
        loss_g = self.adv_weight * self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        loss_model = recon_loss + p_loss + quant_loss + loss_g
 
        logits_fake = self.discriminator(x_recon_logits.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        loss_d = 0.5 * self.adv_weight * (loss_d_fake + loss_d_real)

        log_dict = {
            'valid_loss': loss_model,
            'valid_loss_gen': loss_g,
            'valid_loss_disc': loss_d,
            'valid_loss_recon': recon_loss,
            'valid_loss_quant': quant_loss,
            'valid_loss_percep': p_loss
        }

        self.log_dict(log_dict, batch_size=batch_size, on_step=False, 
                      on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if not self.validation_step_dict:
            self.validation_step_dict['x'] = x
            self.validation_step_dict['x_recon'] = x_recon_logits
            self.validation_step_dict['labels'] = labels
        else:
            self.validation_step_dict['x'] = torch.cat([self.validation_step_dict['x'], x])
            self.validation_step_dict['x_recon'] = torch.cat([self.validation_step_dict['x_recon'], x_recon_logits])
            self.validation_step_dict['labels'] = np.concatenate([self.validation_step_dict['labels'], labels])

        return loss_model
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        x_recon_logits, quant_loss = self.model(x)
        recon_loss = self.recon_weight * F.l1_loss(x_recon_logits.float(), x.float())
        p_loss = self.perceptual_weight * self.perceptual_loss(x_recon_logits.float(), x.float())
        
        logits_fake = self.discriminator(x_recon_logits.contiguous().float())[-1]
        loss_g = self.adv_weight * self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        loss_model = recon_loss + p_loss + quant_loss + loss_g
 
        logits_fake = self.discriminator(x_recon_logits.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        loss_d = 0.5 * self.adv_weight * (loss_d_fake + loss_d_real)

        log_dict = {
            'test_loss': loss_model,
            'test_loss_gen': loss_g,
            'test_loss_disc': loss_d,
            'test_loss_recon': recon_loss,
            'test_loss_quant': quant_loss,
            'test_loss_percep': p_loss
        }

        self.log_dict(log_dict, batch_size=batch_size, on_step=False, 
                      on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if not self.test_step_dict:
            self.test_step_dict['x'] = x
            self.test_step_dict['x_recon'] = x_recon_logits
            self.test_step_dict['labels'] = labels
        else:
            self.test_step_dict['x'] = torch.cat([self.test_step_dict['x'], x])
            self.test_step_dict['x_recon'] = torch.cat([self.test_step_dict['x_recon'], x_recon_logits])
            self.test_step_dict['labels'] = np.concatenate([self.test_step_dict['labels'], labels])

        return loss_model

    def compute_optimization_step(self, model, optimizer, loss):
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        optimizer.step()
        self.untoggle_optimizer(optimizer)

    def on_validation_epoch_end(self) -> None:
        self.validation_step_dict.clear()

    def on_test_epoch_end(self) -> None:
        self.test_step_dict.clear()

    def configure_optimizers(self):
        return [self.optimizer_model, self.optimizer_discriminator], []
