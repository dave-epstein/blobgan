## BlobGAN: Spatially Disentangled Scene Representations<br><sub>Official PyTorch Implementation</sub><br> 

### [Paper](https://arxiv.org/abs/2205.02837) | [Project Page](https://dave.ml/blobgan) | [Video](https://www.youtube.com/watch?v=KpUv82VsU5k) | [Interactive Demo  ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://dave.ml/blobgan/demo)

https://user-images.githubusercontent.com/5674727/168323496-990b46a2-a11d-4192-898a-f5b683d20265.mp4

This repository contains:

* 🚂 Pre-trained BlobGAN models on three datasets: bedrooms, conference rooms, and a combination of kitchens, living rooms, and dining rooms
* 💻 Code based on PyTorch Lightning ⚡ and Hydra 🐍 which fully supports CPU, single GPU, or multi GPU/node training and inference

We also provide an [📓 interactive demo notebook](https://dave.ml/blobgan/demo) to help get started using our model. Download this notebook and run it on your own Python environment, or test it out on Colab. You can:

* 🖌️️ Generate and edit realistic images with an interactive UI
* 📹 Create animated videos showing off your edited scenes

And, **coming soon!**

* 📸 Upload your own image and convert it into blobs!
* 🧬 Programmatically modify images and reproduce results from our paper

## Setup

Run the commands below one at a time to download the latest version of the BlobGAN code, create a Conda environment, and install necessary packages and utilities.

```bash
git clone https://github.com/dave-epstein/blobgan.git
mkdir -p blobgan/logs/wandb
conda create -y -n blobgan python=3.9
conda activate blobgan
conda install -y pytorch=1.11.0 torchvision=0.12.0 torchaudio cudatoolkit=11.3 -c pytorch
conda install -y cudatoolkit-dev=11.3 -c conda-forge
pip install tqdm==4.64.0 hydra-core==1.1.2 omegaconf==2.1.2 clean-fid==0.1.23 wandb==0.12.11 ipdb==0.13.9 lpips==0.1.4 einops==0.4.1 inputimeout==1.0.4 pytorch-lightning==1.5.10 matplotlib==3.5.2 mpl_interactions[jupyter]==0.21.0 protobuf~=3.19.0 moviepy==1.0.3
cd blobgan
python scripts/setup_fid.py
```
And if you haven't installed `ninja` yet on your machine (to compile custom C++ code), do that. On Linux, this looks like:
```
wget -q --show-progress https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
sudo unzip -q ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```


## Running pretrained models

See `scripts/load_model.py` for an example of how to load a pre-trained model (using the provided `load_{blobgan/stylegan}_model` functions, which can be called from elsewhere) and generate images with it. You can also run the file from the command line to generate images and save them to disk. For example, from the `blobgan` directory, you can run:

```bash
python scripts/load_model.py --model_data bed --dl_dir models --save_dir out --n_imgs 32 --save_blobs --label_blobs
```

Or

```bash
python scripts/load_model.py --model_name stylegan --model_data conference --truncate 0.4
```
Note that the first run may take a minute or two longer as custom C++ code is compiled. See the command's help for more details and options: `scripts/load_model.py --help`

## Training your own model

**Before training your model**, you'll need to modify `src/configs/experiments/local.yaml` to include your WandB information and machine-specific configuration (such as path to data -- `dataset.path` or `dataset.basepath` -- and number of GPUs `trainer.gpus`). Alternatively, you can exclude `local` from the `experiments` option in the commands below and specify these parameters directly on the command line. 

To turn off logging entirely, pass `logger=false`, or to only log to disk but not write to server, pass `wandb.offline=true`. Our code currently only supports WandB logging.

Here's an example command which will train a model on LSUN bedrooms. We list the configuration modules to load for this experiment (`blobgan`, `local`, `jitter`) and then specify any other options as we desire. For example, if we wanted to train a model without jitter, we could just remove that module from the `experiments` array.

```bash
python src/run.py +experiment=[blobgan,local,jitter] wandb.name='10-blob BlobGAN on bedrooms' 
```

In some shells, you may need to add extra quotes around some of these options to prevent them from being parsed immediately on the command line.

Train on the LSUN category of your choice by passing in `dataset.category`, e.g. `dataset.category=church`. Tackle multiple categories at once with `dataset=multilsun` and `dataset.categories=[kitchen,bedroom]`.

You can also train on any collection of images by selecting `dataset=imagefolder` and passing in the path. The code expects at least a subfolder named `train` and optional subfolders named `validate` and `test`. The below command also illustrates how to set arbitrary options using Hydra syntax, such as turning off FID logging or changing dataloader batch size:

```bash
python src/run.py +experiment=[blobgan,local,jitter] wandb.name='20-blob BlobGAN on Places' dataset.dataloader.batch_size=24 +model.log_fid_every_epoch=false dataset=imagefolder +dataset.path=/path/to/places/ model.n_features=20
```

Other parameters of interest are likely `trainer.log_every_n_steps`, `model.log_images_every_n_steps`, and `model.log_fid_every_n_steps`, which control frequency of logging scalars, images, and FID (set any of the latter two to -1 to disable). Also check out `checkpoint.every_n_train_steps` and `checkpoint.save_top_k` which dictate checkpoint saving frequency and decide how many most recent checkpoints to keep (`-1` means keep everything).

### Logging FID during training

In the initial codebase setup, you should have run `scripts/setup_fid.py` which will download and install FID statistics for three different datasets:

* Bedrooms: `lsun_bedroom`
* Conference rooms: `lsun_conference`
* Kitchens, living rooms, dining rooms: `lsun_kld`

**If either `model.log_fid_every_n_steps > -1` or `model.log_fid_every_epoch == true`, make sure that `model.fid_stats_name` is passed in.** If you are training on one of the three datasets from the paper, just pass in the string from the list above.

If you are training on your own data, you'll need to first run `setup_fid.py` to precompute statistics on that. The command might look something like:

```bash
python scripts/setup_fid.py --action compute_new --path /path/to/new/data --name newdata -j 32 -bs 256
```

Then, pass `model.fid_stats_name=newdata` on the command line.

### Training StyleGAN2

Many of the above command line options apply (for controlling data and logging). For example, to train a StyleGAN2 model on LSUN conference rooms, run:

```bash
python src/run.py +experiment=[gan,local] wandb.name='Conference room StyleGAN2' dataset.category=conference
```

This uses default StyleGAN2 hyperparameters: R1 regularization on D every 16 steps, path length regularization on G every 4, R1 weight 50 or gamma=100 (the weight is gamma/2).

## Known issues and upcoming functionality
 
* Scripts to reproduce paper figures
* Inversion pretrained models and code

## Citation

If our code or models aided your research, please cite our [paper](https://arxiv.org/abs/2205.02837):
```
@misc{epstein2022blobgan,
      title={BlobGAN: Spatially Disentangled Scene Representations},
      author={Dave Epstein and Taesung Park and Richard Zhang and Eli Shechtman and Alexei A. Efros},
      year={2022},
      eprint={2205.02837},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}		
```

## Code acknowledgments

This repository is built on top of rosinality's excellent [PyTorch re-implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) and Bill Peebles' [GANgealing codebase](https://github.com/wpeebles/gangealing).
