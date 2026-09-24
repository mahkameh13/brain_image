import lightning as L
import torch
import torch.nn.functional as F
from generative.networks.nets import VQVAE as BaseVQVAE
from generative.networks.layers import VectorQuantizer, EMAQuantizer
from omegaconf import OmegaConf


class VQVAE(L.LightningModule):
    def __init__(self, params, optimizer, lr_scheduler):
        super().__init__()
        dict_params = \
            {key: value
             for param in OmegaConf.to_container(params, resolve=True)
             for key, value in param.items()}
        self.model = CustomVQVAE(dict_params)
        self.optim = optimizer(self.parameters())
        self.lr_scheduler = lr_scheduler(self.optim)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        x_recon_logits, quant_loss = self.model(x)
        return x_recon_logits, quant_loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.shape[0]

        x_recon_logits, quant_loss = self(x)
        recon_loss = F.l1_loss(x_recon_logits.float(),
                               x.float(),
                               reduction='sum')
        recon_loss /= batch_size
        quant_loss /= batch_size

        loss = recon_loss + quant_loss

        self.log('train_loss', loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('train_loss_recon', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('train_loss_quant', quant_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        x_recon_logits, quant_loss = self(x)
        recon_loss = F.l1_loss(x_recon_logits.float(),
                               x.float(), reduction='sum')
        recon_loss /= batch_size
        quant_loss /= batch_size

        loss = recon_loss + quant_loss

        self.log('valid_loss', loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('valid_loss_recon', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('valid_loss_quant', quant_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.validation_step_outputs.append((x_recon_logits, labels, None))
        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = x.shape[0]

        x_recon_logits, quant_loss = self(x)
        recon_loss = F.l1_loss(x_recon_logits.float(),
                               x.float(), reduction='sum')
        recon_loss /= batch_size
        quant_loss /= batch_size

        loss = recon_loss + quant_loss

        self.log('test_loss', loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('test_loss_recon', recon_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        self.log('test_loss_quant', quant_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True)
        
        self.test_step_outputs.append((x_recon_logits, labels, None))

        return loss

    def on_validation_epoch_end(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        x, label = batch
        z = self.model.encode(x)
        e, _ = self.model.quantize(z)
        x_recon = self.decode(x)
        return x, x_recon, label, None, None, z

    def configure_optimizers(self):
        return {'optimizer': self.optim,
                'lr_scheduler': self.lr_scheduler,
                'monitor': 'valid_loss'}



class CustomVQVAE(BaseVQVAE):
    def __init__(self, params):
        super().__init__(**params)

        self.quantizer = VectorQuantizer(
            quantizer=CustomEMAQuantizer(params)
        )

class CustomEMAQuantizer(EMAQuantizer):
    def __init__(self, params):
        super().__init__(spatial_dims=params['spatial_dims'],
                         num_embeddings=params['num_embeddings'],
                         embedding_dim=params['embedding_dim'],
                         commitment_cost=params['commitment_cost'],
                         decay=params['decay'],
                         epsilon=params['epsilon'],
                         embedding_init=params['embedding_init'],
                         ddp_sync=params['ddp_sync'])

    def forward(self, inputs):
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                dw = torch.mm(encodings.t(), flat_input)

                if self.ddp_sync:
                    self.distributed_synchronization(encodings_sum, dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
                self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))

        # Encoding Loss
        loss = self.commitment_cost * torch.nn.functional.mse_loss(quantized.detach(),
                                                                   inputs,
                                                                   reduction='sum')
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices