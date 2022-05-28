from __future__ import print_function
import argparse
import os
import numpy as np
import dnnlib.tflib, pickle, torch, collections
from matplotlib import pyplot as plt
import torchvision
from networks.style_gan_net import Generator, BasicDiscriminator
from utils import str2bool
import mlflow

# command line arguments
parser = argparse.ArgumentParser(description='Pytorch style-gan image generation')

parser.add_argument('--convert', type=str2bool,
                    help='convert the official tf checkpoints to my pytorch checkpoints', required=False)

parser.add_argument('--use-official-checkpoints', type=str2bool,
                    help='use official tf checkpoints', required=False)

parser.add_argument('--random-seed', type=int,
                    help='random seed for generating latent', required=False)

parser.add_argument('--dataset', type=str, default='ffhq',
                    help='random seed for generating latent', required=False)

parser.add_argument('--g-checkpoint', type=str,
                    help='generator checkpoint file path')

parser.add_argument('--target-resolution', type=int,
                    help='target resolution. Can be different from trained resolution which is from the checkpoint')

parser.add_argument('--nrow', type=int,
                    help='number of rows for the image grid')

parser.add_argument('--ncol', type=int,
                    help='number of columns for the image grid')

args = parser.parse_args()

def convert(weights, generator, g_out_file, discriminator, d_out_file):
    # get the weights
    weights_pt = [
        collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k, v in w.trainables.items()]) for w in
        weights]
    torch.save(weights_pt, g_out_file)
    # convert
    _G, _D, _Gs = torch.load(g_out_file)
    def key_translate(k, to_print=False):
        k = k.lower().split('/')
        if to_print:
            print(k)
        if k[0] == 'g_synthesis':
            if not k[1].startswith('torgb'):
                k.insert(1, 'blocks')
            if k[1].startswith('torgb'):
                k.insert(1, 'torgbs')
                if k[-1] == 'weight':
                    k.insert(3, 'module')
            k = '.'.join(k)
            k = (k.replace('const.const', 'const')
                 # early block
                 .replace('const.bias', 'bias')
                 .replace('const.noise.weight', 'epi0.pre_style_op.noise.weight')
                 .replace('const.stylemod.weight', 'epi0.style_mod.linear.module.weight')
                 .replace('const.stylemod.bias', 'epi0.style_mod.linear.bias')
                 #.replace('const.stylemod', 'epi0.style_mod.linear')
                 .replace('conv.weight', 'conv.module.weight')
                 .replace('conv.noise.weight', 'epi1.pre_style_op.noise.weight')
                 .replace('conv.stylemod.weight', 'epi1.style_mod.linear.module.weight')
                 .replace('conv.stylemod.bias', 'epi1.style_mod.linear.bias')
                 # later blocks
                 .replace('conv0_up.weight', 'conv0_up.conv.module.weight')
                 .replace('conv0_up.bias', 'conv0_up.conv.bias')
                 .replace('conv0_up.noise.weight', 'epi0.pre_style_op.noise.weight')
                 .replace('conv0_up.stylemod.weight', 'epi0.style_mod.linear.module.weight')
                 .replace('conv0_up.stylemod.bias', 'epi0.style_mod.linear.bias')

                 .replace('conv1.weight', 'conv1.module.weight')
                 #.replace('conv1.bias', 'conv1.module.bias')
                 .replace('conv1.noise.weight', 'epi1.pre_style_op.noise.weight')
                 .replace('conv1.stylemod.weight', 'epi1.style_mod.linear.module.weight')
                 .replace('conv1.stylemod.bias', 'epi1.style_mod.linear.bias')
                 #.replace('torgb_lod0', 'torgb')
                 #.replace('torgb_lod0.weight', 'torgb_lod0.module.weight')
                 #.replace('torgb_lod0.bias', 'torgb_lod0.bias')
                  )
        elif k[0] == 'g_mapping':
            # mapping net
            if k[2] == 'weight':
                k.insert(2, 'module')
            k = '.'.join(k)
        # discriminator
        else:
            if k[0].startswith('fromrgb'):
                k.insert(0, 'fromrgbs')
                if k[-1] == 'weight':
                    k.insert(2, 'module')
            else:
                k.insert(0, 'blocks')
            k = '.'.join(k)
            k = (k
                  #.replace('fromrgb_lod0.weight', 'fromrgb.module.weight')
                  #.replace('fromrgb_lod0.bias', 'fromrgb.bias')
                  .replace('conv0.weight', 'conv0.module.weight')
                  .replace('conv1_down.weight', 'conv1_down.conv.module.weight')
                  .replace('conv1_down.bias', 'conv1_down.conv.bias')
                  .replace('conv.weight', 'conv.module.weight')
                  .replace('dense0.weight', 'dense0.module.weight')
                  .replace('dense1.weight', 'dense1.module.weight')
                 )
        return k

    def weight_translate(k, w):
        k = key_translate(k)
        if k.endswith('.weight'):
            if w.dim() == 2:
                w = w.t()
            elif w.dim() == 1:
                pass
            else:
                assert w.dim() == 4
                w = w.permute(3, 2, 0, 1)
        return w

    # TODO: Note that for training, our structure is fixed. Needs to support growing.
    def translate_checkpoint_with_defined(checkpoint, defined):
        if 0:
            for k, v in checkpoint.items():
                print('original key in checkpoint: ', k)

        checkpoint_pd = {key_translate(k): weight_translate(k, v) for k, v in checkpoint.items()}

        # we delete the useless torgb_(1-9) and fromrgb(1-9) conv filters which are used for growing training
        checkpoint_pd = {k: v for k, v in checkpoint_pd.items()
                         if k not in (['torgb_lod{}'.format(i) for i in range(1, 9)]
                                      + ['fromrgbs.fromrgb_lod{}'.format(i) for i in range(1, 9)])}
        if 0:
            for k, v in checkpoint_pd.items():
                print('checkpoint parameter ', k, v.shape)
        if 1:
            defined_shapes = {k: v.shape for k, v in defined.state_dict().items()}
            param_shapes = {k: v.shape for k, v in checkpoint_pd.items()}

            for k in list(defined_shapes) + list(param_shapes):
                pds = param_shapes.get(k)
                dss = defined_shapes.get(k)
                if pds is None:
                    print("ours only", k, dss)
                elif dss is None:
                    print("theirs only", k, pds)
                elif dss != pds:
                    print("mismatch!", k, pds, dss)
        return checkpoint_pd

    # translate generator
    if generator is not None:
        g_checkpoint_pd = translate_checkpoint_with_defined(checkpoint=_Gs, defined=generator)
        # strict needs to be False for the blur filters
        generator.load_state_dict(g_checkpoint_pd, strict=False)
        torch.save(generator.state_dict(), g_out_file)

    if discriminator is not None:
        d_checkpoint_pd = translate_checkpoint_with_defined(checkpoint=_D, defined=discriminator)
        discriminator.load_state_dict(d_checkpoint_pd, strict=False)
        torch.save(discriminator.state_dict(), d_out_file)


def get_info(file_path):
    info_dict = {}
    parts = os.path.basename(file_path).split('.')
    info_dict['resolution'] = int(parts[1].split('x')[0])
    # alpha is float and has a '.'
    info_dict['alpha'] = float(parts[2] + '.' + parts[3])
    info_dict['cur_nimg'] = int(parts[4])
    info_dict['cur_tick'] = int(parts[5])

    return info_dict


if __name__ == '__main__':
    ##### cmd line arguments #####
    to_convert = args.convert
    use_official_checkpoints = args.use_official_checkpoints
    random_seed = args.random_seed
    nrow = args.nrow
    ncol = args.ncol
    if not use_official_checkpoints:
        g_out_file = args.g_checkpoint
        info = get_info(g_out_file)
        alpha = info['alpha']
        # this is hard coded
        resolution = args.target_resolution
        trained_resolution = info['resolution']
        generator = Generator(resolution=resolution)

    else:
        #############################
        # changes this accordingly
        dataset = args.dataset
        checkpoint_prefix = 'kerras2019stylegan'
        alpha = 1.0
        url = {'cats': ('https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ',
                        256,
                        '1c6bbbad79102cf05f29ec0363071cf3'),
               'bedrooms': ('https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF',
                            256,
                            '258371067819a08c899eeb7d1d2c8c19'
                            ),
               'ffhq': ('https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
                        1024,
                        '263e666dc20e26dcbfa514733c1d1f81'
                        )
               }[dataset]

        pretrained_dir = 'pretrained'
        resolution = url[1]
        g_out_file = os.path.join(pretrained_dir,
                                  'karras2019stylegan-{}-{}x{}.generator.pt'.format(dataset, resolution, resolution))
        generator = Generator(resolution=resolution)

        discriminator = None
        d_out_file = None
        trained_resolution = resolution
        if 0:
            d_out_file = os.path.join(pretrained_dir,
                                    'karras2019stylegan-{}-{}x{}.discriminator.pt'.format(dataset, resolution, resolution))
        if 0:
            discriminator = BasicDiscriminator(resolution=resolution)

        if to_convert:
            # init tf
            print('start conversion')
            dnnlib.tflib.init_tf()
            try:
                with dnnlib.util.open_url(url[0], cache_dir=pretrained_dir) as f:
                    weights = pickle.load(f)

            except:
                weights = pickle.load(open(os.path.join(pretrained_dir,
                                                        'karras2019stylegan-{}-{}x{}.pkl'.format(dataset, resolution, resolution))))
            convert(weights,
                    generator=generator,
                    g_out_file=g_out_file,
                    discriminator=discriminator,
                    d_out_file=d_out_file)

            print('finished conversion')

    with mlflow.start_run():

        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        generator.load_state_dict(torch.load(g_out_file))

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        generator.eval()
        generator.to(device)
        torch.manual_seed(random_seed)

        resolution_log2 = int(np.log2(trained_resolution))

        latents = torch.randn(nrow * ncol, 512, device=device)
        with torch.no_grad():
            # alpha is 1
            imgs = generator(latents, resolution_log2, alpha)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0


        output_imgs = imgs
        imgs = torchvision.utils.make_grid(output_imgs.cpu(), nrow=ncol, normalize=True)

        # torchvision.utils.save_image(imgs, 'sample_00.png', padding=2, nrow=nrow, normalize=True)
        plt.figure(figsize=(30, 12))
        #if use_official_checkpoints:
        imgs = imgs.permute(1, 2, 0)
        plt.imshow(imgs.detach().numpy())
        plt.show()

        if 0:
            discriminator.load_state_dict(torch.load(d_out_file))

            discriminator.eval()
            discriminator.to(device)
            # alpha is 1
            result = discriminator(output_imgs, resolution_log2, 1).cpu().detach().numpy()
            print(result)


