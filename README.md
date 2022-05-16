<<<<<<< HEAD
## BlobGAN: Spatially Disentangled Scene Representations<br><sub>Official PyTorch Implementation</sub><br> 

### [Paper](https://arxiv.org/abs/2205.02837) | [Project Page](https://dave.ml/blobgan) | [Video](https://www.youtube.com/watch?v=KpUv82VsU5k) | [Interactive Demo  ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://dave.ml/blobgan/demo)

https://user-images.githubusercontent.com/5674727/168323496-990b46a2-a11d-4192-898a-f5b683d20265.mp4

This repository contains:

* 🚂 Pre-trained BlobGAN models on three datasets: bedrooms, conference rooms, and a combination of kitchens, living rooms, and dining rooms
* 💻 Code based on PyTorch Lightning ⚡ and Hydra 🐍 which fully supports CPU, single GPU, or multi GPU/node training and inference

And, **coming soon**, easy-to-run 🖋scripts to:

* 🖌️️ Generate and edit realistic images with an interactive UI
* 📸 Upload your own image and convert it into blobs!
* 🧬 Programmatically modify images and reproduce results from our paper

## Setup

Run the commands below one at a time to download the latest version of the BlobGAN code, create a Conda environment, and install necessary packages and utilities.

```bash
git clone https://github.com/dave-epstein/blobgan.git
mkdir -p blobgan/logs/wandb
conda create -n blobgan python=3.9
conda activate blobgan
conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio cudatoolkit=11.3 -c pytorch
conda install cudatoolkit-dev=11.3 -c conda-forge
pip install tqdm==4.64.0 hydra-core==1.1.2 omegaconf==2.1.2 clean-fid==0.1.23 wandb==0.12.11 ipdb==0.13.9 lpips==0.1.4 einops==0.4.1 inputimeout==1.0.4 pytorch-lightning==1.5.10 matplotlib==3.5.2 mpl_interactions[jupyter]==0.21.0
wget -q --show-progress https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
sudo unzip -q ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
````


## Running pretrained models (coming very soon!)

See `scripts/load_model.py` for an example of how to load a pre-trained model and generate images with it. For example:

```bash
python scripts/load_model.py --model_name bed --dl_dir models --save_dir out --n_imgs 32 --save_blobs --label_blobs
```

See the command's help for more details and options: `scripts/load_model.py --help`

## Training your own model (coming very soon!)

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
=======
# Official PyTorch implementation of BlobGAN: Spatially Disentangled Scene Representations

More details coming soon! In the meantime, please check out our [interactive notebook (run locally or on Colab)](https://dave.ml/blobgan/demo).
>>>>>>> 886a44bbc329932c391f357c76d365522dc741ba
