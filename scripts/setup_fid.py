import os
import subprocess
import sys
import cleanfid
import torch
import numpy as np
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_folder_features
from torchvision.transforms import functional as F

here_dir = os.path.dirname(__file__)

sys.path.append(os.path.join(here_dir, '..', 'src'))
os.environ['PYTHONPATH'] = os.path.join(here_dir, '..', 'src')

from utils import download


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--action', default='download', choices=['download', 'compute_new'],
                        help='All other options only apply if action is set to `compute_new`.'
                             ' Download mode (default) simply configures precomputed stats used in the BlobGAN paper.')
    parser.add_argument('--path', default='', type=str,
                        help='Path to custom folder from which to sample `--n_imgs` images and compute FID statistics.')
    parser.add_argument('--n_imgs', type=int, default=-1,
                        help='Number of images to randomly sample for FID stats. Set to -1 to use all images.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the files in a directory before selecting `--n_imgs` for computation.')
    parser.add_argument('--name', default=None, help='Name to give custom stats.')
    parser.add_argument('-bs', '--batch_size', default=32,
                        help='Number of images to analyze in one forward pass. Adjust based on available GPU memory.',
                        type=int)
    parser.add_argument('-j', '--num_workers', default=8,
                        help='Number of workers to use for FID stats generation.',
                        type=int)
    parser.add_argument('-r', '--resolution', default=256,
                        help='Image resolution to use before feeding images into FID pipeline (where they are resized to 299).',
                        type=int)
    parser.add_argument('--device', default='cuda',
                        help='Specify the device on which to run the code, in PyTorch syntax, '
                             'e.g. `cuda`, `cpu`, `cuda:3`.')
    args = parser.parse_args()


    def load_fn(x):
        x = F.resize(torch.from_numpy(x).permute(2, 0, 1), args.resolution)
        x = F.center_crop(x, args.resolution).permute(1, 2, 0)
        return np.array(x)

    if args.action == 'download':
        path = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
        stats = download(path=path, file='fid_stats.tar.gz')
        subprocess.run(["tar", "xvzf", stats, '-C', path, '--strip-components', '1'])
    else:
        print('Calculating...')
        name, mode, device, fdir, num = args.name, "clean", torch.device(args.device), args.path, args.n_imgs
        assert name
        stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
        os.makedirs(stats_folder, exist_ok=True)
        split, res = "custom", "na"
        outname = f"{name}_{mode}_{split}_{res}.npz"
        outf = os.path.join(stats_folder, outname)
        # if the custom stat file already exists
        if os.path.exists(outf):
            msg = f"The statistics file {name} already exists. "
            msg += "Use remove_custom_stats function to delete it first."
            raise Exception(msg)

        feat_model = build_feature_extractor(mode, device)
        fbname = os.path.basename(fdir)
        # get all inception features for folder images
        if num < 0: num = None
        np_feats = get_folder_features(fdir, feat_model, num_workers=args.num_workers, num=num, shuffle=args.shuffle,
                                       batch_size=args.batch_size, device=device, custom_image_tranform=load_fn,
                                       mode=mode, description=f"custom stats: {fbname} : ")
        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        print(f"Saving custom FID stats to {outf}")
        np.savez_compressed(outf, mu=mu, sigma=sigma)
