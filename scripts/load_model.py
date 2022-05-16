from pathlib import Path
import os, sys
import torch
from PIL import Image
from tqdm import tqdm

here_dir = os.path.dirname(__file__)

sys.path.append(os.path.join(here_dir, '../src'))
sys.path.append(os.path.join(here_dir, '../src/blobgan'))
os.environ['PYTHONPATH'] = os.path.join(here_dir, '../src/blobgan')

from models import BlobGAN
from utils import download_model, download_mean_latent, download_cherrypicked, KLD_COLORS, BED_CONF_COLORS, \
    viz_score_fn, for_canvas, draw_labels


def load_model(model_name, path, device='cuda'):
    ckpt = download_model(model_name, path)
    model = BlobGAN.load_from_checkpoint(ckpt, strict=False).to(device)
    model.mean_latent = download_mean_latent(model_name, path).to(device)
    model.cherry_picked = download_cherrypicked(model_name, path).to(device)
    COLORS = KLD_COLORS if 'kitchen' in model_name else BED_CONF_COLORS
    model.colors = COLORS
    noise = [torch.randn((1, 1, 16 * 2 ** ((i + 1) // 2), 16 * 2 ** ((i + 1) // 2))).to(device) for i in
             range(model.generator_ema.num_layers)]
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
    print('\033[92m' 'Done loading and configuring model!', flush=True)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default='bed',
                        help="Choose a pretrained model. This must be a string that begins either with `bed` (bedrooms),"
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
    parser.add_argument("--save_blobs", action='store_true', help='If passed, save images of blob maps.')
    parser.add_argument("--label_blobs", action='store_true',
                        help='If passed, add numeric blob labels to blob map images, when `--save_blobs` is true.')
    parser.add_argument('--size_threshold', default=-3,
                        help='Threshold for blob size parameter above which to render blob labels, when `--label_blobs` is true.',
                        type=float)
    parser.add_argument('--device', default='cuda',
                        help='Specify the device on which to run the code, in PyTorch syntax, e.g. `cuda`, `cpu`, `cuda:3`.')
    args = parser.parse_args()

    model = load_model(args.model_name, args.dl_dir)

    save_dir = Path(args.save_dir)
    (save_dir / 'imgs').mkdir(exist_ok=True)
    if args.save_blobs:
        (save_dir / 'blobs').mkdir(exist_ok=True)
        if args.label_blobs:
            (save_dir / 'blobs_labeled').mkdir(exist_ok=True)

    n_to_gen = args.n_imgs
    n_gen = 0

    with tqdm(total=args.n_imgs) as pbar:
        while n_to_gen > 0:
            bs = min(args.batch_size, n_to_gen)
            z = torch.randn((bs, 512)).to(args.device)

            layout, orig_img = model.gen(z=z, truncate=args.truncate, **model.render_kwargs)

            for i in range(len(orig_img)):
                img_i = for_canvas(orig_img[i:i + 1])
                Image.fromarray(img_i).save(str(save_dir / 'imgs' / f'{i + n_gen:04d}.png'))
                if args.save_blobs:
                    blobs_i = for_canvas(layout['feature_img'][i:i + 1].mul(255))
                    Image.fromarray(blobs_i).save(str(save_dir / 'blobs' / f'{i + n_gen:04d}.png'))
                    if args.label_blobs:
                        labeled_blobs, labeled_blobs_img = draw_labels(blobs_i, layout, args.size_threshold,
                                                                       model.colors, layout_i=i)
                        labeled_blobs_img.save(str(save_dir / 'blobs_labeled' / f'{i + n_gen:04d}.png'))

            n_to_gen -= bs
            n_gen += bs
            pbar.update(bs)
