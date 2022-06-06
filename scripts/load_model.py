from pathlib import Path
import os, sys
import torch
from PIL import Image
from tqdm import tqdm, trange

here_dir = os.path.dirname(__file__)

sys.path.append(os.path.join(here_dir, '..', 'src'))
os.environ['PYTHONPATH'] = os.path.join(here_dir, '..', 'src')

from models import BlobGAN, GAN
from utils import download_model, download_mean_latent, download_cherrypicked, KLD_COLORS, BED_CONF_COLORS, \
    viz_score_fn, for_canvas, draw_labels, download


def load_SGAN1_bedrooms(path, device='cuda'):
    ckpt = download(path=path, file='SGAN1_bedrooms.ckpt', load=True)
    sys.path.append(os.path.join(here_dir, 'style-gan-pytorch'))
    from networks.style_gan_net import Generator

    model = Generator(resolution=256)
    model.load_state_dict(ckpt)
    model.eval()
    return model.to(device)


def load_stylegan_model(model_data, path, device='cuda'):
    if model_data.startswith('bed'):
        model = load_SGAN1_bedrooms(path, device)
        Z = torch.randn((10000, 512)).to(device)
        latents = [model.g_mapping(Z[_:_ + 1])[0] for _ in trange(10000, desc='Computing mean latent')]
        model.mean_latent = torch.stack(latents, 0).mean(0)

        def SGAN1_gen(z, truncate):
            a = 1 - truncate
            dlatents = model.g_mapping(z).clone()
            if a < 1:
                dlatents = a * dlatents + (1 - a) * model.mean_latent
            x = model.g_synthesis(dlatents, 8, 1).clone()
            xx = ((x.clamp(min=-1, max=1) + 1) / 2.0) * 255
            return xx

        model.gen = SGAN1_gen
    else:
        datastr = 'conference' if model_data.startswith('conference') else 'kitchenlivingdining'
        ckpt = download(path=path, file=f'SGAN2_{datastr}.ckpt')
        model = GAN.load_from_checkpoint(ckpt, strict=False).to(device)
        model.get_mean_latent()

        def SGAN2_gen(z, truncate):
            return model.generator_ema([z], return_image_only=True, truncation=1 - truncate,
                                       truncation_latent=model.mean_latent).add_(1).div_(2).mul_(255)

        model.gen = SGAN2_gen
    return model


def load_blobgan_model(model_data, path, device='cuda', fixed_noise=False):
    ckpt = download_model(model_data, path)
    model = BlobGAN.load_from_checkpoint(ckpt, strict=False).to(device)
    try:
        model.mean_latent = download_mean_latent(model_data, path).to(device)
    except:
        model.get_mean_latent()
    try:
        model.cherry_picked = download_cherrypicked(model_data, path).to(device)
    except:
        pass
    COLORS = KLD_COLORS if 'kitchen' in model_data else BED_CONF_COLORS
    model.colors = COLORS
    noise = [torch.randn((1, 1, 16 * 2 ** ((i + 1) // 2), 16 * 2 ** ((i + 1) // 2))).to(device) for i in
             range(model.generator_ema.num_layers)] if fixed_noise else None
    model.noise = noise
    render_kwargs = {
        'no_jitter': True,
        'ret_layout': True,
        'viz': True,
        'ema': True,
        'viz_colors': COLORS,
        'norm_img': True,
        'viz_score_fn': viz_score_fn,
        'noise': noise
    }
    model.render_kwargs = render_kwargs
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_name", default='blobgan', choices=['blobgan', 'stylegan'])
    parser.add_argument("-d", "--model_data", default='bed',
                        help="Choose a pretrained model. This must be a string that begins either with `bed_no_jitter` (bedrooms, trained without jitter), "
                             "`bed` (bedrooms),"
                             " `kitchen` (kitchens, living rooms, and dining rooms),"
                             " or `conference` (conference rooms).")
    parser.add_argument("-dl", "--dl_dir", default='models',
                        help='Path to a directory where model files will be downloaded.')
    parser.add_argument("-s", "--save_dir", default='out',
                        help='Path to the directory where output images will be saved.')
    parser.add_argument("-n", "--n_imgs", default=100, type=int, help='Number of random images to generate.')
    parser.add_argument('-bs', '--batch_size', default=32,
                        help='Number of images to generate in one forward pass. Adjust based on available GPU memory.',
                        type=int)
    parser.add_argument('-t', '--truncate', default=0.4,
                        help='Amount of truncation to use when generating images. 0 means no truncation, 1 means full truncation.',
                        type=float)
    parser.add_argument("--save_blobs", action='store_true',
                        help='If passed, save images of blob maps (when `--model_name` is BlobGAN).')
    parser.add_argument("--label_blobs", action='store_true',
                        help='If passed, add numeric blob labels to blob map images, when `--save_blobs` is true.')
    parser.add_argument('--size_threshold', default=-3,
                        help='Threshold for blob size parameter above which to render blob labels, when `--label_blobs` is true.',
                        type=float)
    parser.add_argument('--device', default='cuda',
                        help='Specify the device on which to run the code, in PyTorch syntax, e.g. `cuda`, `cpu`, `cuda:3`.')
    args = parser.parse_args()

    blobgan = args.model_name == 'blobgan'

    save_dir = Path(args.save_dir)
    (save_dir / 'imgs').mkdir(exist_ok=True, parents=True)

    if blobgan:
        model = load_blobgan_model(args.model_data, args.dl_dir, args.device)

        if args.save_blobs:
            (save_dir / 'blobs').mkdir(exist_ok=True, parents=True)
            if args.label_blobs:
                (save_dir / 'blobs_labeled').mkdir(exist_ok=True, parents=True)
    else:
        model = load_stylegan_model(args.model_data, args.dl_dir, args.device)

    n_to_gen = args.n_imgs
    n_gen = 0

    torch.set_grad_enabled(False)

    with tqdm(total=args.n_imgs, desc='Generating images') as pbar:
        while n_to_gen > 0:
            bs = min(args.batch_size, n_to_gen)
            z = torch.randn((bs, 512)).to(args.device)

            if blobgan:
                layout, orig_img = model.gen(z=z, truncate=args.truncate, **model.render_kwargs)
            else:
                orig_img = model.gen(z=z, truncate=args.truncate)

            for i in range(len(orig_img)):
                img_i = for_canvas(orig_img[i:i + 1])
                Image.fromarray(img_i).save(str(save_dir / 'imgs' / f'{i + n_gen:04d}.png'))
                if blobgan and args.save_blobs:
                    blobs_i = for_canvas(layout['feature_img'][i:i + 1].mul(255))
                    Image.fromarray(blobs_i).save(str(save_dir / 'blobs' / f'{i + n_gen:04d}.png'))
                    if args.label_blobs:
                        labeled_blobs, labeled_blobs_img = draw_labels(blobs_i, layout, args.size_threshold,
                                                                       model.colors, layout_i=i)
                        labeled_blobs_img.save(str(save_dir / 'blobs_labeled' / f'{i + n_gen:04d}.png'))

            n_to_gen -= bs
            n_gen += bs
            pbar.update(bs)
