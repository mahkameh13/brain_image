import torch
import torch.nn.functional as F

from generative.losses.perceptual import medicalnet_intensity_normalisation, normalize_tensor
from generative.metrics.fid import get_fid_score
from generative.metrics.mmd import MMDMetric
from generative.metrics.ms_ssim import MultiScaleSSIMMetric
from generative.metrics.ssim import SSIMMetric
from monai.metrics.regression import PSNRMetric
from monai.metrics.metric import Metric


class FID(Metric):
    def __init__(self, net='medicalnet_resnet10_23datasets', verbose=False):
        super().__init__()

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        self.model = torch.hub.load("Warvito/MedicalNet-models",
                                    model=net,
                                    verbose=verbose)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, x, x_recon):
        assert len(x.shape) == len(x_recon.shape) == 5

        self.model = self.model.to(device=x.device, dtype=x.dtype)

        norm_x = medicalnet_intensity_normalisation(x)
        norm_x_recon = medicalnet_intensity_normalisation(x_recon)

        features_x = self.model.forward(norm_x)
        features_x_recon = self.model.forward(norm_x_recon)

        norm_features_x = normalize_tensor(features_x)
        norm_features_x_recon = normalize_tensor(features_x_recon)

        avg_norm_features_x = norm_features_x.mean([2, 3, 4])
        avg_norm_features_x_recon = norm_features_x_recon.mean([2, 3, 4])

        try:
            fid_score = get_fid_score(avg_norm_features_x,
                                      avg_norm_features_x_recon)
        except ValueError:
            fid_score = 1000

        return fid_score


class MMD(Metric):
    def __init__(self, **params):
        super().__init__()
        self.metric = MMDMetric(**params)

    def __call__(self, x, x_recon):
        mmd = self.metric(x, x_recon)
        return mmd


class MSSSIM(Metric):
    def __init__(self, **params):
        super().__init__()
        self.metric = MultiScaleSSIMMetric(**params)

    def __call__(self, x, x_recon):
        msssim_score = self.metric(x_recon, x)

        return msssim_score.mean()


class MAE(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, x, x_recon):
        mae = F.l1_loss(x, x_recon)
        return mae


class MSE(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, x, x_recon):
        mse = F.mse_loss(x, x_recon)
        return mse


class PSNR(Metric):
    def __init__(self, **params):
        super().__init__()
        self.metric = PSNRMetric(**params)

    def __call__(self, x, x_recon):
        psnr = self.metric(x_recon, x)

        return psnr.mean()


class SSIM(Metric):
    def __init__(self, **params):
        super().__init__()
        self.metric = SSIMMetric(**params)

    def __call__(self, x, x_recon):
        ssim_score = self.metric(x_recon, x)

        return ssim_score.mean()
