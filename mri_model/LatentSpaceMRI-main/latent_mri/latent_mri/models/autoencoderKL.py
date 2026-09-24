import lightning as L
import torch
import torch.nn.functional as F
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL as MonaiAutoencoderKL
from generative.networks.nets import PatchDiscriminator
from omegaconf import OmegaConf


class AutoencoderKL(L.LightningModule):
    def __init__(self, params, optimizer):
        super().__init__()
        dict_params_autoencoder = \
            {key: value
             for param in OmegaConf.to_container(params['autoencoder'],
                                                 resolve=True)
             for key, value in param.items()}
        dict_params_discriminator = \
            {key: value
             for param in OmegaConf.to_container(params['discriminator'],
                                                 resolve=True)
             for key, value in param.items()}
        dict_params_loss = \
            {key: value
             for param in OmegaConf.to_container(params['loss'],
                                                 resolve=True)
             for key, value in param.items()}

        self.model = MonaiAutoencoderKL(**dict_params_autoencoder)
        self.discriminator = PatchDiscriminator(**dict_params_discriminator)

        self.optimizer_model = \
            optimizer['autoencoder'](self.model.parameters())
        self.optimizer_discriminator = \
            optimizer['discriminator'](self.discriminator.parameters())

        self.perceptual_loss = PerceptualLoss(spatial_dims=3,
                                              network_type="squeeze",
                                              fake_3d_ratio=0.25)
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        self.adv_weight = dict_params_loss['adv_weight']
        self.perceptual_weight = dict_params_loss['perceptual_weight']
        self.kl_weight = dict_params_loss['kl_weight']

        self.validation_step_dict = {}
        self.test_step_dict = {}
        self.save_hyperparameters()
        self.automatic_optimization = False

    def forward(self, x):
        reconstruction, z_mu, z_sigma = self.model(x)
        return reconstruction, z_mu, z_sigma

    def calculate_generator_loss(self, x, x_recon_logits, z_mu, z_sigma):
        logits_fake = \
            self.discriminator(x_recon_logits.contiguous().float())[-1]

        recon_loss = F.l1_loss(x_recon_logits.float(), x.float())
        p_loss = self.perceptual_loss(x_recon_logits.float(), x.float())
        generator_loss = self.adv_loss(logits_fake,
                                       target_is_real=True,
                                       for_discriminator=False)

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) -
                                  torch.log(z_sigma.pow(2)) - 1,
                                  dim=[1, 2, 3, 4])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        return recon_loss, kl_loss, p_loss, generator_loss

    def calculate_discriminator_loss(self, x, x_recon_logits):
        logits_fake = \
            self.discriminator(x_recon_logits.contiguous().detach())[-1]
        d_loss_fake = self.adv_loss(logits_fake, target_is_real=False,
                                    for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        d_loss_real = self.adv_loss(logits_real, target_is_real=True,
                                    for_discriminator=True)
        d_loss = (d_loss_fake + d_loss_real) * 0.5

        return d_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]

        optimizer_model, optimizer_discriminator = self.optimizers()

        x_recon_logits, z_mu, z_sigma = self.model(x)

        recon_loss, kl_loss, p_loss, g_loss = \
            self.calculate_generator_loss(x, x_recon_logits, z_mu, z_sigma)
        total_g_loss = recon_loss + \
            (self.kl_weight * kl_loss) + \
            (self.perceptual_weight * p_loss) + \
            (self.adv_weight * g_loss)
        self.compute_optimization_step(optimizer_model, total_g_loss)

        d_loss = \
            self.calculate_discriminator_loss(x, x_recon_logits)
        total_d_loss = self.adv_weight * d_loss
        self.compute_optimization_step(optimizer_discriminator, total_d_loss)

        train_loss = total_g_loss + total_d_loss
        log_dict = {
            'train_loss': train_loss,
            'train_loss_generator': g_loss,
            'train_loss_discriminator': d_loss,
            'train_loss_recon': recon_loss,
            'train_kl_loss': kl_loss,
            'train_perceptual_loss': p_loss
        }

        self.log_dict(log_dict, batch_size=batch_size, on_step=False,
                      on_epoch=True, prog_bar=True, logger=True,
                      sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        x_recon_logits, z_mu, z_sigma = self.model(x)

        recon_loss, kl_loss, p_loss, g_loss = \
            self.calculate_generator_loss(x, x_recon_logits, z_mu, z_sigma)
        total_g_loss = recon_loss + \
            (self.kl_weight * kl_loss) + \
            (self.perceptual_weight * p_loss) + \
            (self.adv_weight * g_loss)

        d_loss = \
            self.calculate_discriminator_loss(x, x_recon_logits)
        total_d_loss = self.adv_weight * d_loss

        valid_loss = total_g_loss + total_d_loss
        log_dict = {
            'valid_loss': valid_loss,
            'valid_loss_generator': g_loss,
            'valid_loss_discriminator': d_loss,
            'valid_loss_recon': recon_loss,
            'valid_kl_loss': kl_loss,
            'valid_perceptual_loss': p_loss
        }

        self.log_dict(log_dict, batch_size=batch_size, on_step=False,
                      on_epoch=True, prog_bar=True, logger=True,
                      sync_dist=True)

        if not self.validation_step_dict:
            self.validation_step_dict['x'] = x
            self.validation_step_dict['x_recon'] = x_recon_logits
            self.validation_step_dict['labels'] = labels
        else:
            self.validation_step_dict['x'] = \
                torch.cat([self.validation_step_dict['x'], x])
            self.validation_step_dict['x_recon'] = \
                torch.cat([self.validation_step_dict['x_recon'],
                           x_recon_logits])
            self.validation_step_dict['labels'] = \
                torch.cat([self.validation_step_dict['labels'], labels])

        return valid_loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        x_recon_logits, z_mu, z_sigma = self.model(x)

        recon_loss, kl_loss, p_loss, g_loss = \
            self.calculate_generator_loss(x, x_recon_logits, z_mu, z_sigma)
        total_g_loss = recon_loss + \
            (self.kl_weight * kl_loss) + \
            (self.perceptual_weight * p_loss) + \
            (self.adv_weight * g_loss)

        d_loss = \
            self.calculate_discriminator_loss(x, x_recon_logits)
        total_d_loss = self.adv_weight * d_loss

        test_loss = total_g_loss + total_d_loss
        log_dict = {
            'test_loss': test_loss,
            'test_loss_generator': g_loss,
            'test_loss_discriminator': d_loss,
            'test_loss_recon': recon_loss,
            'test_kl_loss': kl_loss,
            'test_perceptual_loss': p_loss
        }

        self.log_dict(log_dict, batch_size=batch_size, on_step=False,
                      on_epoch=True, prog_bar=True, logger=True,
                      sync_dist=True)

        if not self.test_step_dict:
            self.test_step_dict['x'] = x
            self.test_step_dict['x_recon'] = x_recon_logits
            self.test_step_dict['labels'] = labels
        else:
            self.test_step_dict['x'] = \
                torch.cat([self.test_step_dict['x'], x])
            self.test_step_dict['x_recon'] = \
                torch.cat([self.test_step_dict['x_recon'],
                           x_recon_logits])
            self.test_step_dict['labels'] = \
                torch.cat([self.test_step_dict['labels'], labels])

        return test_loss

    def compute_optimization_step(self, optimizer, loss):
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
