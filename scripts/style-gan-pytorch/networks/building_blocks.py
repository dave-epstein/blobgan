import torch
import torch.nn as nn
from collections import OrderedDict
from networks.custom_layers import *


class LayerEpilogue(nn.Module):
    """
    Things to do at the end of each layer
    1. mixin scaled noise
    2. mixin style with AdaIN
    """
    def __init__(self,
                 num_channels,
                 dlatent_size,        # Disentangled latent (W) dimensionality,
                 use_wscale,         # Enable equalized learning rate?
                 use_pixel_norm,    # Enable pixel-wise feature vector normalization?
                 use_instance_norm,
                 use_noise,
                 use_styles,
                 nonlinearity,
                 ):
        super(LayerEpilogue, self).__init__()

        act = {
               'relu': torch.relu,
               'lrelu': nn.LeakyReLU(negative_slope=0.2)
               }[nonlinearity]

        layers = []
        if use_noise:
            layers.append(('noise', NoiseMixin(num_channels)))
        layers.append(('act', act))

        # to follow the tf implementation
        if use_pixel_norm:
            layers.append(('pixel_norm', NormalizationLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(num_channels)))
        # now we need to mixin styles
        self.pre_style_op = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMixin(dlatent_size,
                                        num_channels,
                                        use_wscale=use_wscale)
    def forward(self, x, dlatent):
        # dlatent is w
        x = self.pre_style_op(x)
        if self.style_mod:
            x = self.style_mod(x, dlatent)
        return x


class EarlySynthesisBlock(nn.Module):
    """
    The first block for 4x4 resolution
    """
    def __init__(self,
                 in_channels,
                 dlatent_size,
                 const_input_layer,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles,
                 nonlinearity
                 ):
        super(EarlySynthesisBlock, self).__init__()
        self.const_input_layer = const_input_layer
        self.in_channels = in_channels

        if const_input_layer:
            self.const = nn.Parameter(torch.ones(1, in_channels, 4, 4))
            self.bias = nn.Parameter(torch.ones(in_channels))
        else:
            self.dense = EqualizedLinear(dlatent_size, in_channels * 16, use_wscale=use_wscale)

        self.epi0 = LayerEpilogue(num_channels=in_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_noise=use_noise,
                                  use_pixel_norm=use_pixel_norm,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity
                                  )
        # kernel size must be 3 or other odd numbers
        # so that we have 'same' padding
        self.conv = EqualizedConv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=3,
                                    padding=3//2)

        self.epi1 = LayerEpilogue(num_channels=in_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_noise=use_noise,
                                  use_pixel_norm=use_pixel_norm,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity
                                  )

    def forward(self, dlatents):
        # note dlatents is broadcast one
        dlatents_0 = dlatents[:, 0]
        dlatents_1 = dlatents[:, 1]
        batch_size = dlatents.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_0).view(batch_size, self.in_channels, 4, 4)

        x = self.epi0(x, dlatents_0)
        x = self.conv(x)
        x = self.epi1(x, dlatents_1)
        return x


class LaterSynthesisBlock(nn.Module):
    """
    The following blocks for res 8x8...etc.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles,
                 nonlinearity,
                 blur_filter,
                 res,
                 ):
        super(LaterSynthesisBlock, self).__init__()

        # res = log2(H), H is 4, 8, 16, 32 ... 1024

        assert isinstance(res, int) and (2 <= res <= 10)

        self.res = res

        if blur_filter:
            self.blur = Blur2d(blur_filter)
            #blur = Blur2d(blur_filter)
        else:
            self.blur = None

        # name 'conv0_up' is used in tf implementation
        self.conv0_up = Upscale2dConv2d(res=res,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        use_wscale=use_wscale)
       # self.conv0_up = Upscale2dConv2d2(
       #     input_channels=in_channels,
       #     output_channels=out_channels,
       #     kernel_size=3,
       #     gain=np.sqrt(2),
       #     use_wscale=use_wscale,
       #     intermediate=blur,
       #     upscale=True
       # )

        self.epi0 = LayerEpilogue(num_channels=out_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_pixel_norm=use_pixel_norm,
                                  use_noise=use_noise,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity)

        # name 'conv1' is used in tf implementation
        # kernel size must be 3 or other odd numbers
        # so that we have 'same' padding
        # no upsclaing
        self.conv1 = EqualizedConv2d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     padding=3//2)

        self.epi1 = LayerEpilogue(num_channels=out_channels,
                                  dlatent_size=dlatent_size,
                                  use_wscale=use_wscale,
                                  use_pixel_norm=use_pixel_norm,
                                  use_noise=use_noise,
                                  use_instance_norm=use_instance_norm,
                                  use_styles=use_styles,
                                  nonlinearity=nonlinearity)


    def forward(self, x, dlatents):

        x = self.conv0_up(x)
        if self.blur is not None:
            x = self.blur(x)
        x = self.epi0(x, dlatents[:, self.res * 2 - 4])
        x = self.conv1(x)
        x = self.epi1(x, dlatents[:, self.res * 2 - 3])
        return x


class EarlyDiscriminatorBlock(nn.Sequential):
    def __init__(self,
                 res,
                 in_channels,
                 out_channels,
                 use_wscale,
                 blur_filter,
                 fused_scale,
                 nonlinearity):
        act = {
            'relu': torch.relu,
            'lrelu': nn.LeakyReLU(negative_slope=0.2)
        }[nonlinearity]

        layers = []

        layers.append(('conv0', EqualizedConv2d(in_channels=in_channels,
                                                out_channels=in_channels,
                                                kernel_size=3,
                                                padding=3//2,
                                                use_wscale=use_wscale)))
        # note that we don't have layer epilogue in discriminator, so we need to add activation layer mannually
        layers.append(('act0', act))

        layers.append(('blur', Blur2d(blur_filter)))

        layers.append(('conv1_down', Downscale2dConv2d(res=res,
                                                       in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=3,
                                                       fused_scale=fused_scale,
                                                       use_wscale=use_wscale)))
        layers.append(('act1', act))

        super().__init__(OrderedDict(layers))


class LaterDiscriminatorBlock(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_wscale,
                 nonlinearity,
                 mbstd_group_size,
                 mbstd_num_features,
                 res,
                 ):
        act = {
            'relu': torch.relu,
            'lrelu': nn.LeakyReLU(negative_slope=0.2)
        }[nonlinearity]

        resolution = 2 ** res
        layers = []
        layers.append(('minibatchstddev', MiniBatchStdDev(mbstd_group_size, mbstd_num_features)))
        layers.append(('conv', EqualizedConv2d(in_channels=in_channels + mbstd_num_features,
                                               out_channels=in_channels,
                                               kernel_size=3,
                                               padding=3//2,
                                               use_wscale=use_wscale)))
        layers.append(('act0', act))
        layers.append(('flatten', Flatten()))
        layers.append(('dense0', EqualizedLinear(in_channels=in_channels * (resolution**2),
                                                  out_channels=in_channels,
                                                  use_wscale=use_wscale)))
        layers.append(('act1', act))
        # no activation for the last fc
        layers.append(('dense1', EqualizedLinear(in_channels=in_channels,
                                                 out_channels=out_channels)))

        super().__init__(OrderedDict(layers))
