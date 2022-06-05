from __future__ import annotations

__all__ = ["BlobGANInverter"]

import random
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple

import torch
import torch.optim as optim
from PIL import Image
from lpips import LPIPS
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.optim import Optimizer
from torchvision.utils import make_grid

from models import networks, BlobGAN
from models.base import BaseModule
from utils import FromConfig, run_at_step, freeze, is_rank_zero, load_pretrained_weights, to_dataclass_cfg, print_once

# SPLAT_KEYS = ['spatial_style', 'xs', 'ys', 'covs', 'sizes']
SPLAT_KEYS = ['spatial_style', 'scores_pyramid']
_ = Image
_ = make_grid


@dataclass
class Lossλs:
    real_LPIPS: float = 1.
    real_MSE: float = 1.
    fake_LPIPS: float = 1.
    fake_MSE: float = 1.
    fake_latents_MSE: float = 1.

    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass(eq=False)
class BlobGANInverter(BaseModule):
    # Modules
    inverter: FromConfig[nn.Module]
    generator: FromConfig[BlobGAN]
    # Loss parameters
    λ: FromConfig[Lossλs] = None
    # Logging
    log_images_every_n_steps: Optional[int] = 500
    log_timing_every_n_steps: Optional[int] = -1
    log_grads_every_n_steps: Optional[int] = -1
    valtest_log_all: bool = False
    # Resuming
    generator_pretrained: Optional[Union[str, DictConfig]] = None
    # Optim
    lr: float = 0.002
    eps: float = 1e-5
    # Training
    trunc_min: float = 0.0
    trunc_max: float = 0.0

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()
        cfg = to_dataclass_cfg(self.generator, BlobGAN)
        if self.generator_pretrained.log_dir is None:
            self.generator_pretrained.log_dir = 'logs/'
        self.generator = load_pretrained_weights('BlobGAN', self.generator_pretrained, BlobGAN(**cfg), strict=False)
        del self.generator.discriminator
        del self.generator.generator
        del self.generator.layout_net
        freeze(self.generator)
        self.inverter = networks.get_network(**self.inverter,
                                             d_out=self.generator.layout_net_ema.mlp[-1].weight.shape[0])
        self.L_LPIPS = LPIPS(net='vgg', verbose=False)
        freeze(self.L_LPIPS)
        self.λ = Lossλs(**self.λ)

    # Initialization and state management
    def on_train_start(self):
        super().on_train_start()
        # Validate parameters w.r.t. trainer (must be done here since trainer is not attached as property yet in init)
        assert self.log_images_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder. ' \
            f'Got {self.log_images_every_n_steps} and {self.trainer.log_every_n_steps}.'
        assert self.log_timing_every_n_steps < 0 or self.log_timing_every_n_steps % self.trainer.log_every_n_steps == 0, \
            '`model.log_images_every_n_steps` must be divisible by `trainer.log_every_n_steps` without remainder'

    def configure_optimizers(self) -> Union[optim, List[optim]]:
        params = list(self.inverter.parameters())
        print_once(f'Optimizing {sum([p.numel() for p in params]) / 1e6:.2f}M params')
        return torch.optim.AdamW(params, lr=self.lr, eps=self.eps, weight_decay=0)

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ):
        self.batch_idx = batch_idx
        optimizer.step(closure=optimizer_closure)

    def shared_step(self, batch: Tuple[Tensor, dict], batch_idx: int,
                    optimizer_idx: Optional[int] = None, mode: str = 'train') -> Optional[Union[Tensor, dict]]:
        """
        Args:
            batch: tuple of tensor of shape N x C x H x W of images and a dictionary of batch metadata/labels
            batch_idx: pytorch lightning training loop batch index
            optimizer_idx: pytorch lightning optimizer index (0 = G, 1 = D)
            mode:
                `train` returns the total loss and logs losses and images/profiling info.
                `validate`/`test` log total loss and return images
        Returns: see description for `mode` above
        """
        # Set up modules and data
        batch_real, batch_labels = batch
        log_images = run_at_step(self.trainer.global_step, self.log_images_every_n_steps)

        z = torch.randn(len(batch_real), self.generator.noise_dim).type_as(batch_real)

        with torch.no_grad():
            truncate = self.trunc_min if self.trunc_min == self.trunc_max \
                else random.uniform(self.trunc_min, self.trunc_max)
            layout_gt_fake, gen_imgs = self.generator.gen(z, truncate=truncate, ema=True, viz=log_images,
                                                          ret_layout=True)

        losses = dict()

        z_pred_fake = self.inverter(gen_imgs.detach())

        layout_pred_fake, reconstr_fake = self.generator.gen(z_pred_fake, ema=True, viz=log_images, ret_layout=True,
                                                             mlp_idx=len(self.generator.layout_net_ema.mlp))

        losses['fake_MSE'] = (gen_imgs - reconstr_fake).pow(2).mean()
        losses['fake_LPIPS'] = self.L_LPIPS(reconstr_fake, gen_imgs).mean()
        latent_l2_loss = []
        for k in ('xs', 'ys', 'covs', 'sizes', 'features', 'spatial_style'):
            latent_l2_loss.append((layout_pred_fake[k] - layout_gt_fake[k].detach()).pow(2).mean())
        losses['fake_latents_MSE'] = sum(latent_l2_loss) / len(latent_l2_loss)

        layout_pred_real, reconstr_real = self.generator.gen(z_pred_fake, ema=True, viz=log_images, ret_layout=True,
                                                             mlp_idx=len(self.generator.layout_net_ema.mlp))

        losses['real_MSE'] = (batch_real - reconstr_real).pow(2).mean()
        losses['real_LPIPS'] = self.L_LPIPS(reconstr_real, batch_real).mean()

        total_loss = f'total_loss'
        losses[total_loss] = sum(map(lambda k: losses[k] * self.λ[k], losses))
        isnan = self.alert_nan_loss(losses[total_loss], batch_idx)
        if self.all_gather(isnan).any():
            if self.ipdb_on_nan and is_rank_zero():
                import ipdb
                ipdb.set_trace()
            return
        self.log_scalars(losses, mode)

        imgs = {
            'real': batch_real,
            'real_reconstr': reconstr_real,
            'fake': gen_imgs,
            'fake_reconstr': reconstr_fake,
            'real_reconstr_feats': layout_pred_real['feature_img'],
            'fake_reconstr_feats': layout_pred_fake['feature_img'],
            'fake_feats': layout_gt_fake['feature_img']
        }
        if mode == "train":
            if log_images and is_rank_zero():
                imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items()}
                self._log_image_dict(imgs, mode, square_grid=False, ncol=len(batch_real))
            return losses[total_loss]
        else:
            if self.valtest_log_all:
                imgs = self.gather_tensor_dict(imgs)
            return imgs
