import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.nets import Discriminator
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from omegaconf import OmegaConf
from torch.autograd import Variable


class VAEGAN(L.LightningModule):
    def __init__(self, beta, gamma, params, optimizer):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        dict_params_vae = \
            {key: value
             for param in OmegaConf.to_container(params['vae'], resolve=True)
             for key, value in param.items()}
        dict_params_discriminator = \
            {key: value
             for param in OmegaConf.to_container(params['discriminator'],
                                                 resolve=True)
             for key, value in param.items()}
        self.encoder = Encoder(**dict_params_vae)
        self.decoder = Decoder(**dict_params_vae)
        self.discriminator = Discriminator(**dict_params_discriminator)

        self.optimizer_encoder = optimizer['encoder'](self.encoder.parameters())
        self.optimizer_decoder = optimizer['decoder'](self.decoder.parameters())
        self.optimizer_discriminator = \
            optimizer['discriminator'](self.discriminator.parameters())

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.discriminator_activations = \
            self.get_activations(self.discriminator)

    @property
    def model(self):
        return VAE(self.encoder, self.decoder)

    def forward(self, inputs):
        z, mu, logvar = self.encoder(inputs)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def predict(self, inputs):
        z, mu, logvar = self.encoder(inputs)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]

        optimizer_encoder, optimizer_decoder, optimizer_discriminator = \
            self.optimizers()

        loss, _ = self.compute_forward_pass(x)

        loss_encoder, loss_decoder, loss_disc, loss_disc_x, loss_disc_recon, loss_disc_noise, loss_mse = loss

        loss_encoder /= batch_size
        loss_decoder /= batch_size
        loss_disc /= batch_size

        train_loss = loss_encoder + loss_decoder + loss_disc

        self.compute_optimization_step(optimizer_encoder, loss_encoder)
        self.compute_optimization_step(optimizer_decoder, loss_decoder)
        self.compute_optimization_step(optimizer_discriminator,
                                       loss_disc)

        self.log('train_loss', train_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('train_loss_encoder', loss_encoder, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('train_loss_decoder', loss_decoder, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('train_loss_disc', loss_disc,
                 batch_size=batch_size, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True)
        self.log('beta', self.beta, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('gamma', self.gamma, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        
        loss_disc_x /= batch_size
        loss_disc_recon /= batch_size
        loss_disc_noise /= batch_size
        loss_mse /= batch_size

        self.log('train_loss_disc_x', loss_disc_x, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('train_loss_disc_recon', loss_disc_recon, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('train_loss_disc_noise', loss_disc_noise, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('train_loss_mse', loss_mse, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        loss, predictions = self.compute_forward_pass(x)

        loss_encoder, loss_decoder, loss_disc, loss_disc_x, loss_disc_recon, loss_disc_noise, loss_mse = loss

        loss_encoder /= batch_size
        loss_decoder /= batch_size
        loss_disc /= batch_size

        valid_loss = loss_encoder + loss_decoder + loss_disc

        x_recon_logits, z = predictions

        self.log('valid_loss', valid_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('valid_loss_encoder', loss_encoder, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('valid_loss_decoder', loss_decoder, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('valid_loss_disc', loss_disc,
                 batch_size=batch_size, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True)
        
        loss_disc_x /= batch_size
        loss_disc_recon /= batch_size
        loss_disc_noise /= batch_size
        loss_mse /= batch_size


        self.log('valid_loss_disc_x', loss_disc_x, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('valid_loss_disc_recon', loss_disc_recon, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('valid_loss_disc_noise', loss_disc_noise, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('valid_loss_mse', loss_mse, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)

        self.validation_step_outputs.append((x_recon_logits, labels, z))

        return valid_loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        loss, predictions = self.compute_forward_pass(x)

        loss_encoder, loss_decoder, loss_disc, loss_disc_x, loss_disc_recon, loss_disc_noise, loss_mse = loss

        loss_encoder /= batch_size
        loss_decoder /= batch_size
        loss_disc /= batch_size

        test_loss = loss_encoder + loss_decoder + loss_disc

        x_recon_logits, z = predictions

        self.log('test_loss', test_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('test_loss_encoder', loss_encoder, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('test_loss_decoder', loss_decoder, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('test_loss_disc', loss_disc,
                 batch_size=batch_size, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, sync_dist=True)
        
        loss_disc_x /= batch_size
        loss_disc_recon /= batch_size
        loss_disc_noise /= batch_size
        loss_mse /= batch_size


        self.log('test_loss_disc_x', loss_disc_x, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('test_loss_disc_recon', loss_disc_recon, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('test_loss_disc_noise', loss_disc_noise, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)
        self.log('test_loss_mse', loss_mse, batch_size=batch_size,
                on_step=False, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True)

        self.test_step_outputs.append((x_recon_logits, labels, z))

        return test_loss

    def predict_step(self, batch, batch_idx):
        x, label = batch
        x_recon_logits, mu, logvar, z = self(x)
        x_recon = F.sigmoid(x_recon_logits)
        return x, x_recon, label, mu, logvar, z

    def on_validation_epoch_end(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return [self.optimizer_encoder, self.optimizer_decoder,
                self.optimizer_discriminator], []

    def compute_forward_pass(self, x):
        batch_size = x.shape[0]
        z, mu, logvar = self.encoder(x)
        loss_kl = self.loss_kl(mu, logvar)
        x_recon_logits = self.decoder(z)

        out_disc_x = self.discriminator(x)
        activations_disc_x = self.discriminator_activations['layer_3']
        out_disc_recon = self.discriminator(F.sigmoid(x_recon_logits))
        activations_disc_recon = self.discriminator_activations['layer_3']
        loss_recon = self.loss_recon(activations_disc_x,
                                     activations_disc_recon)
        loss_mse = self.loss_mse(x, F.sigmoid(x_recon_logits))

        noise = torch.randn(batch_size, 1, 1, self.encoder.latent_size)
        noise = noise.type_as(x)
        noise.requires_grad = True
        noise_recon = self.decoder(noise)
        out_disc_noise = self.discriminator(F.sigmoid(noise_recon))
        loss_disc = self.loss_discriminator(out_disc_x,
                                            out_disc_recon,
                                            out_disc_noise,
                                            batch_size)
        loss_disc_x, loss_disc_recon, loss_disc_noise = loss_disc
        loss_disc_total = loss_disc_x + loss_disc_recon + loss_disc_noise

        loss_encoder = self.beta * loss_kl + loss_recon
        loss_decoder = self.gamma * loss_recon - loss_disc_total

        loss = (loss_encoder, loss_decoder, loss_disc_total, 
                loss_disc_x, loss_disc_recon, loss_disc_noise, loss_mse)
        predictions = (x_recon_logits, z)
        return loss, predictions

    def compute_optimization_step(self, optimizer, loss):
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        optimizer.step()
        self.untoggle_optimizer(optimizer)

    def loss_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

    def loss_mse(self, x, x_recon):
        return F.mse_loss(x_recon, x, reduction='sum')

    def loss_recon(self, activations_x, activations_recon):
        return torch.sum((activations_x - activations_recon) ** 2)

    def _loss_discriminator(self, out_disc_x, out_disc_recon, out_disc_noise,
                           batch_size):
        label_x = torch.ones(batch_size, 1).type_as(out_disc_x)
        label_recon = torch.zeros(batch_size, 1).type_as(out_disc_recon)
        label_noise = torch.zeros(batch_size, 1).type_as(out_disc_noise)

        label_x.requires_grad = True
        label_recon.requires_grad = True
        label_noise.requires_grad = True

        loss_disc_x = F.binary_cross_entropy_with_logits(out_disc_x,
                                                    label_x,
                                                    reduction='sum')
        loss_disc_recon = F.binary_cross_entropy_with_logits(out_disc_recon,
                                                        label_recon,
                                                        reduction='sum')
        loss_disc_noise = F.binary_cross_entropy_with_logits(out_disc_noise,
                                                        label_noise,
                                                        reduction='sum')

        return loss_disc_x, loss_disc_recon, loss_disc_noise
    
    def loss_discriminator(self, out_disc_x, out_disc_recon, out_disc_noise,
                           batch_size):
        out_disc_x = F.sigmoid(out_disc_x)
        out_disc_recon = F.sigmoid(out_disc_recon)
        out_disc_noise = F.sigmoid(out_disc_noise)

        label_x = torch.ones(batch_size, 1).type_as(out_disc_x)
        label_recon = torch.zeros(batch_size, 1).type_as(out_disc_recon)
        label_noise = torch.zeros(batch_size, 1).type_as(out_disc_noise)

        # label_x -= 0.2 * torch.rand_like(label_x).type_as(label_x)
        # label_recon += 0.2 * torch.rand_like(label_recon).type_as(label_recon)
        # label_noise += 0.2 * torch.rand_like(label_noise).type_as(label_noise)

        label_x.requires_grad = True
        label_recon.requires_grad = True
        label_noise.requires_grad = True

        loss_disc_x = F.mse_loss(out_disc_x, label_x, reduction='sum')
        loss_disc_recon = F.mse_loss(out_disc_recon, label_recon, reduction='sum')
        loss_disc_noise = F.mse_loss(out_disc_noise, label_noise, reduction='sum')

        return loss_disc_x, loss_disc_recon, loss_disc_noise

    def get_activations(self, model):
        activation = {}

        def register_activations(name):
            def hook(model, input, output):
                activation[name] = output  # output.detach()
            return hook
        layer = model.net.layer_3
        layer.register_forward_hook(register_activations('layer_3'))
        return activation


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def decode_forward(self, z, use_sigmoid):
        x = F.relu(self.decoder.decodeL(z))
        x = x.view(x.shape[0], self.decoder.channels[-1], *self.decoder.final_size)
        if use_sigmoid:
            return F.sigmoid(self.decoder(x))
        return self.decoder(x)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


class Encoder(nn.Module):
    def __init__(self, spatial_dims, latent_size, in_shape, in_channels,
                 out_channels, channels, strides, kernel_size, up_kernel_size,
                 act, norm, dropout, bias):
        super().__init__()
        self.dimensions = spatial_dims
        self.latent_size = latent_size
        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        # Check the number of channels and strides match
        if len(channels) != len(strides):
            raise ValueError('Encoder expects matching number of channels'
                             'and strides')
        self.final_size = np.asarray(self.in_shape, dtype=int)
        padding = same_padding(self.kernel_size)
        for stride in self.strides:
            self.final_size = calculate_out_shape(self.final_size,
                                                  self.kernel_size,
                                                  stride,
                                                  padding)

        self.encoder, self.encoded_channels = self._get_encoder()
        self.linear_size = int(np.prod(self.final_size)) * \
            self.encoded_channels
        self.mu = nn.Linear(self.linear_size, self.latent_size)
        self.logvar = nn.Linear(self.linear_size, self.latent_size)

    def _get_encoder(self):
        encoder = nn.Sequential()
        layer_channels = self.in_channels
        for i, (channels, strides) in \
                enumerate(zip(self.channels, self.strides)):
            layer = self._get_encoder_layer(layer_channels,
                                            channels,
                                            strides,
                                            False)
            encoder.add_module(f'encoder_{i}', layer)
            layer_channels = channels

        return encoder, channels

    def _get_encoder_layer(self, in_channels, out_channels, strides, is_last):
        return Convolution(spatial_dims=self.dimensions,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           strides=strides,
                           kernel_size=self.kernel_size,
                           act=self.act,
                           norm=self.norm,
                           dropout=self.dropout,
                           bias=self.bias,
                           padding=None,
                           conv_only=is_last)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        if self.training:
            std = torch.randn_like(std).mul(std)

        return std.add_(mu)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, spatial_dims, latent_size, in_shape, in_channels,
                 out_channels, channels, strides, kernel_size, up_kernel_size,
                 act, norm, dropout, bias):
        super().__init__()
        self.dimensions = spatial_dims
        self.latent_size = latent_size
        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        # Check the number of channels and strides match
        if len(channels) != len(strides):
            raise ValueError('Encoder expects matching number of channels'
                             'and strides')
        self.final_size = np.asarray(self.in_shape, dtype=int)
        padding = same_padding(self.kernel_size)
        for stride in self.strides:
            self.final_size = calculate_out_shape(self.final_size,
                                                  self.kernel_size,
                                                  stride,
                                                  padding)
        self.decoder = self._get_decoder()
        self.linear_size = int(np.prod(self.final_size)) * \
            self.channels[-1]
        self.decodeL = nn.Linear(self.latent_size, self.linear_size)

    def _get_decoder(self):
        decoder = nn.Sequential()
        list_channels = self.channels[-2::-1] + [self.out_channels]
        list_strides = self.strides[::-1] or [1]

        layer_channels = self.channels[-1]
        for i, (channels, strides) in \
                enumerate(zip(list_channels, list_strides)):
            layer = self._get_decoder_layer(layer_channels,
                                            channels,
                                            strides,
                                            i == (len(list_strides) - 1))
            decoder.add_module(f'decoder_{i}', layer)
            layer_channels = channels

        return decoder

    def _get_decoder_layer(self, in_channels, out_channels, strides, is_last):
        return Convolution(spatial_dims=self.dimensions,
                           in_channels=in_channels,
                           out_channels=out_channels,
                           strides=strides,
                           kernel_size=self.kernel_size,
                           act=self.act,
                           norm=self.norm,
                           dropout=self.dropout,
                           bias=self.bias,
                           padding=None,
                           conv_only=is_last,
                           is_transposed=True)

    def forward(self, x):
        x = F.relu(self.decodeL(x))
        x = x.view(x.shape[0], self.channels[-1], *self.final_size)
        return self.decoder(x)
