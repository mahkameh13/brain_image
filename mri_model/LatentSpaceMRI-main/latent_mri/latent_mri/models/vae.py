import lightning as L
import torch
import torch.nn.functional as F
from monai.networks.nets import VarAutoEncoder
from omegaconf import OmegaConf


class VAE(L.LightningModule):
    def __init__(self, beta, params, optimizer, lr_scheduler):
        super().__init__()
        self.beta = beta
        dict_params = \
            {key: value
             for param in OmegaConf.to_container(params, resolve=True)
             for key, value in param.items()}
        self.model = VarAutoEncoder(**dict_params)
        self.optim = optimizer(self.parameters())
        self.lr_scheduler = lr_scheduler(self.optim)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, inputs):
        x_recon, mu, logvar, z = self.model(inputs)
        return x_recon, mu, logvar, z

    def predict(self, inputs):
        x_recon, mu, logvar, z = self.model(inputs)
        return x_recon, mu, logvar, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]
        x_recon_logits, mu, logvar, _ = self(x)

        recon_loss, kld_loss = self.loss(x, x_recon_logits, mu, logvar)
        recon_loss /= batch_size
        kld_loss = self.beta * kld_loss / batch_size
        train_loss = recon_loss + kld_loss

        self.log('train_loss', train_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('train_bce', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('train_kld', kld_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('beta', self.beta, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]
        x_recon_logits, mu, logvar, z = self(x)

        recon_loss, kld_loss = self.loss(x, x_recon_logits, mu, logvar)
        recon_loss /= batch_size
        kld_loss = self.beta * kld_loss / batch_size
        valid_loss = recon_loss + kld_loss

        self.log('valid_loss', valid_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('valid_bce', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('valid_kld', kld_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.validation_step_outputs.append((x_recon_logits, labels, z))
        return valid_loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]
        x_recon_logits, mu, logvar, z = self(x)

        recon_loss, kld_loss = self.loss(x, x_recon_logits, mu, logvar)
        recon_loss /= batch_size
        kld_loss = self.beta * kld_loss / batch_size
        test_loss = recon_loss + kld_loss

        self.log('test_loss', test_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)        # return bce + kld
        self.log('test_bce', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('test_kld', kld_loss, batch_size=batch_size,
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
        x_recon_logits, mu, logvar, z = self(x)
        x_recon = F.sigmoid(x_recon_logits)
        return x, x_recon, label, mu, logvar, z

    def configure_optimizers(self):
        return {'optimizer': self.optim,
                'lr_scheduler': self.lr_scheduler,
                'monitor': 'valid_loss'}

    def loss(self, x, x_recon_logits, mu, logvar):
        recon_loss = \
            F.binary_cross_entropy_with_logits(x_recon_logits, x,
                                               reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld_loss
