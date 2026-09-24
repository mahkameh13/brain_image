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


class Autoencoder(L.LightningModule):
    def __init__(self, params, optimizer, lr_scheduler):
        super().__init__()
        dict_params = \
            {key: value
             for param in OmegaConf.to_container(params, resolve=True)
             for key, value in param.items()}
        self.model = AE(dict_params)
        self.optim = optimizer(self.parameters())
        self.lr_scheduler = lr_scheduler(self.optim)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, inputs):
        x_recon, z = self.model(inputs)
        return x_recon, z

    def predict(self, inputs):
        x_recon, z = self.model(inputs)
        return x_recon, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]
        x_recon_logits, _ = self(x)

        recon_loss = self.loss(x, x_recon_logits)
        recon_loss /= batch_size
        train_loss = recon_loss

        self.log('train_loss', train_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('train_bce', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]
        x_recon_logits, z = self(x)

        recon_loss = self.loss(x, x_recon_logits)
        recon_loss /= batch_size
        valid_loss = recon_loss

        self.log('valid_loss', valid_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('valid_bce', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)

        self.validation_step_outputs.append((x_recon_logits, labels, z))
        return valid_loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]
        x_recon_logits, z = self(x)

        recon_loss = self.loss(x, x_recon_logits)
        recon_loss /= batch_size
        
        test_loss = recon_loss

        self.log('test_loss', test_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)        # return bce + kld
        self.log('test_bce', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)

        self.test_step_outputs.append((x_recon_logits, labels, z))
        return test_loss

    def on_validation_epoch_end(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        x, label = batch
        x_recon_logits, z = self(x)
        x_recon = F.sigmoid(x_recon_logits)
        return x, x_recon, label, z

    def configure_optimizers(self):
        return {'optimizer': self.optim,
                'lr_scheduler': self.lr_scheduler,
                'monitor': 'valid_loss'}
    
    


    def loss(self, x, x_recon_logits):
        recon_loss = \
            F.binary_cross_entropy_with_logits(x_recon_logits, x,
                                               reduction='sum')
        return recon_loss


class AE(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = Encoder(**params)
        self.decoder = Decoder(**params)

    def decode_forward(self, z, use_sigmoid):
        x_recon = self.decoder(z)
        if use_sigmoid:
            return F.sigmoid(x_recon)
        return x_recon

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class Encoder(nn.Module):
    def __init__(self, spatial_dims, latent_size, in_shape, in_channels,
                 out_channels, channels, strides, kernel_size, up_kernel_size,
                 act, norm, dropout, bias):
        super().__init__()
        self.spatial_dims = spatial_dims
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
        self.encodeL = nn.Linear(self.linear_size, self.latent_size)

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
        return Convolution(spatial_dims=self.spatial_dims,
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

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], self.linear_size)
        z = self.encodeL(x)
        return z


class Decoder(nn.Module):
    def __init__(self, spatial_dims, latent_size, in_shape, in_channels,
                 out_channels, channels, strides, kernel_size, up_kernel_size,
                 act, norm, dropout, bias):
        super().__init__()
        self.spatial_dims = spatial_dims
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
        return Convolution(spatial_dims=self.spatial_dims,
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
