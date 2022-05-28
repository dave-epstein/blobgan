import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import prod


class NormalizationLayer(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

def _upscale2d(x, factor):
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x

class Upscale2d(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """

    def __init__(self, factor=2):
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        return _upscale2d(x, self.factor)


class Downscale2d(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """
    def __init__(self, factor=2):
        super(Downscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        factor = self.factor
        if factor == 1:
            return x
        return F.avg_pool2d(x, factor)


class Blur2d(nn.Module):
    """
    Note: no weight needed for this class
    It's just convenient to define it as a module subclass
    """
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(Blur2d, self).__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x

class MiniBatchStdDev(nn.Module):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input
    Args:
        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """

    def __init__(self, subgroup_size, num_features):
        super(MiniBatchStdDev, self).__init__()
        self.subgroup_size=subgroup_size
        self.num_features=num_features

    def forward(self, x):

        s = x.size()
        subgroup_size = min(s[0], self.subgroup_size)
        if s[0] % subgroup_size != 0:
            subgroup_size = s[0]
        if subgroup_size > 1:
            y = x.view(subgroup_size, -1, self.num_features, s[1]//self.num_features, s[2], s[3])
            y = y - y.mean(0, keepdim=True)
            y = (y ** 2).mean(0, keepdim=True)
            y = (y + 1e-8) ** 0.5
            y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
            y = y.expand(subgroup_size, -1, -1, s[2], s[3]).contiguous().reshape(s[0], self.num_features, s[2], s[3])
        else:
            y = torch.zeros(x.size(0), self.num_features, x.size(2), x.size(3), device=x.device)

        return torch.cat([x, y], dim=1)


def getLayerNormalizationFactor(x, gain):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return gain * math.sqrt(1.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 use_wscale=True,
                 lrmul=1.0,
                 bias=True,
                 gain=np.sqrt(2)):
        r"""
        use_wscale (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        init_bias_to_zero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = use_wscale

        if bias:
            # size(0) is num_out_channels
            self.bias = torch.nn.Parameter(torch.zeros(self.module.weight.size(0)))
            self.bias_mul = 1.0
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrmul
            # this is the multiplier that are used for equalized learning rate
            self.weight_mul = getLayerNormalizationFactor(self.module, gain=gain) * lrmul
            self.bias_mul = lrmul

    def forward(self, x):

        # hack hack. module's bias is always false
        x = self.module(x)
        if self.equalized:
            # this is different from the tf implementation!
            x *= self.weight_mul
        # add on bias
        if self.bias is not None:
            if x.dim() == 2:
                x = x + self.bias.view(1, -1) * self.bias_mul
            else:
                x = x + self.bias.view(1, -1, 1, 1) * self.bias_mul
        return x


class Flatten(nn.Module):
    # for connect conv2d and linear

    def forward(self, x):
        return x.view(x.size(0), -1)

class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            in_channels (int): number of channels in the previous layer
            out_channels (int): number of channels of the current layer
            kernel_size (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """

        # always set bias to False
        # and apply bias manually in constrained layer
        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(in_channels, out_channels,
                                            kernel_size, padding=padding,
                                            bias=False),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            in_channels (int): number of channels in the previous layer
            out_channels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(in_channels, out_channels,
                                            bias=False), **kwargs)


class SmoothUpsample(nn.Module):
    """
    https://arxiv.org/pdf/1904.11486.pdf
    'Making Convolutional Networks Shift-Invariant Again'
    # this is in the tf implementation too
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super(SmoothUpsample, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels,
                                               in_channels,
                                               kernel_size,
                                               kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # change to in_channels, out_channels, kernel_size, kernel_size
        weight = self.weight.permute([1, 0, 2, 3])
        weight = F.pad(weight, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:]
                  + weight[:, :, :-1, 1:]
                  + weight[:, :, 1:, :-1]
                  + weight[:, :, :-1, :-1]
                 )
        x = F.conv_transpose2d(x,
                               weight,
                               self.bias, # note if bias set to False, this will be None
                               stride=2,
                               padding=self.padding)
        return x

# TODO: this needs to be better wrappered by ConstrainedLayer for bias
class EqualizedSmoothUpsample(ConstrainedLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 **kwargs):
        ConstrainedLayer.__init__(self, SmoothUpsample(in_channels,
                                                       out_channels,
                                                       kernel_size=kernel_size,
                                                       bias=False), **kwargs)


class SmoothDownsample(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True
                 ):
        super(SmoothDownsample, self).__init__()
        # weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        weight = F.pad(self.weight, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:]
                  + weight[:, :, :-1, 1:]
                  + weight[:, :, 1:, :-1]
                  + weight[:, :, :-1, :-1]
                 ) / 4
        x = F.conv2d(x, weight, stride=2, padding=self.padding)
        return x

# TODO: this needs to be better wrappered by ConstrainedLayer for bias
class EqualizedSmoothDownsample(ConstrainedLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 **kwargs):
        ConstrainedLayer.__init__(self, SmoothDownsample(in_channels,
                                                         out_channels,
                                                         kernel_size=kernel_size,
                                                         bias=False), **kwargs)
class Upscale2dConv2d(nn.Module):

    def __init__(self,
                 res,  # this is used  for determin the fused_scale
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_wscale,
                 fused_scale='auto',
                 **kwargs):
        super(Upscale2dConv2d, self).__init__()
        # kernel_size assert (from official tf implementation):
        # this is due to the fact that the input size is always even
        # and use kernel_size // 2 ensures 'same' padding from tf
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.fused_scale = fused_scale
        self.upscale = Upscale2d()

        if self.fused_scale == 'auto':
            self.fused_scale = (2 ** res) >= 128

        if not self.fused_scale:

            self.conv = EqualizedConv2d(in_channels,
                                        out_channels,
                                        kernel_size,
                                        padding=kernel_size // 2,
                                        use_wscale=use_wscale,
                                        )
        else:
            self.conv = EqualizedSmoothUpsample(in_channels,
                                                out_channels,
                                                kernel_size,
                                                use_wscale=use_wscale,
                                                )

    def forward(self, x):
        if not self.fused_scale:
            return self.conv(self.upscale(x))
        else:
            return self.conv(x)


class Downscale2dConv2d(nn.Module):

    def __init__(self,
                 res,  # this is used  for determin the fused_scale
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_wscale,
                 fused_scale,
                 **kwargs):
        super(Downscale2dConv2d, self).__init__()
        # kernel_size assert (from official tf implementation):
        # this is due to the fact that the input size is always even
        # and use kernel_size // 2 ensures 'same' padding from tf
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert fused_scale in [True, False, 'auto']
        self.fused_scale = fused_scale
        self.downscale = Downscale2d()

        if self.fused_scale == 'auto':
            self.fused_scale = (2 ** res) >= 128

        if not self.fused_scale:

            self.conv = EqualizedConv2d(in_channels,
                                        out_channels,
                                        kernel_size,
                                        # SAME padding
                                        padding=kernel_size // 2,
                                        use_wscale=use_wscale,
                                        )
        else:
            self.conv = EqualizedSmoothDownsample(in_channels,
                                                  out_channels,
                                                  kernel_size,
                                                  use_wscale=use_wscale,
                                                  )

    def forward(self, x):
        if not self.fused_scale:
            return self.conv(self.downscale(x))
        else:
            return self.conv(x)


class NoiseMixin(nn.Module):
    """
    Add noise with channel wise scaling factor
    reference: apply_noise in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """
    def __init__(self, num_channels):
        super(NoiseMixin, self).__init__()
        # initialize with 0's
        # per-channel scaling factor
        # 'B' in the paper
        # use weight to match the tf implementation
        self.weight = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, noise=None):
        # NCHW
        assert len(x.size()) == 4
        s = x.size()
        if noise is None:
            noise = torch.randn(s[0], 1, s[2], s[3], device=x.device, dtype=x.dtype)
        x = x + self.weight.view(1, -1, 1, 1) * noise

        return x


class StyleMixin(nn.Module):
    """
    Style modulation.
    reference: style_mod in https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
    """
    def __init__(self,
                 dlatent_size,          # Disentangled latent (W) dimensionality
                 num_channels,
                 use_wscale        # use equalized learning rate?
                 ):
        super(StyleMixin, self).__init__()
        # gain is 1.0 here
        self.linear = EqualizedLinear(dlatent_size, num_channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, w):
        # x is instance normalized
        # w is mapped latent
        style = self.linear(w)
        # style's shape (N, 2 * 512)
        # reshape to (y_s, y_b)
        # style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        # NCHW
        # according to the paper, shape of style (y) would be (N, 2, 512, 1, 1)
        # so shape of y_s is (N, 512, 1, 1)
        # channel-wise, y_s is just a scalar
        shape = [-1, 2, x.size(1)] + [1] * (x.dim() - 2)
        style = style.view(shape)
        return x * (style[:, 0] + 1.) + style[:, 1]




