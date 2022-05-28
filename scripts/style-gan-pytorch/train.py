from __future__ import print_function
import os
import argparse
import time
import numpy as np
import pathlib
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dnnlib import EasyDict
from networks.style_gan_net import Generator, BasicDiscriminator
from loss_criterions.gradient_losses import logisticGradientPenalty
from loss_criterions.base_loss_criterions import Logistic
from utils import str2bool
import mlflow

# command line arguments
parser = argparse.ArgumentParser(description='Pytorch style-gan training')

parser.add_argument('--data-root', type=str, help='image data root directory')

parser.add_argument('--resume', type=str2bool, nargs='?', help='resume training')

parser.add_argument('--g-checkpoint', type=str,
                    help='generator checkpoint path for continuing training when resume is set to True')
parser.add_argument('--d-checkpoint', type=str,
                    help='discriminator checkpoint path for continuing training when resume is set to True')

parser.add_argument('--target-resolution', type=int, help='target resolution for training (default: 128)')

parser.add_argument('--n-gpu', type=int, help='number of gpus for training (default: 1)')

args = parser.parse_args()


real_label = 1
fake_label = 0


def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def get_resume_info_from_checkpoint(g_checkpoint,
                                    d_checkpoint
                                    ):

    def get_info(file_path):
        info_dict = {}
        parts = os.path.basename(file_path).split('.')
        info_dict['resolution'] = int(parts[1].split('x')[0])
        # alpha is float and has a '.'
        info_dict['alpha'] = float(parts[2] + '.' + parts[3])
        info_dict['cur_nimg'] = int(parts[4])
        info_dict['cur_tick'] = int(parts[5])

        return info_dict

    g_info = get_info(g_checkpoint)
    d_info = get_info(d_checkpoint)
    return g_info, d_info




def training_schedule(cur_nimg,
                      resolution_log2,                   # current resolution_log2
                      num_gpus,
                      lod_initial_resolution=4,
                      lod_training_kimg=600,
                      lod_transition_kimg=600,
                      minibatch_base=4,
                      minibatch_dict={},
                      max_minibatch_per_gpu = {},
                      G_lrate_base=0.001,
                      G_lrate_dict={},
                      D_lrate_base=0.001,
                      D_lrate_dict={},
                      lrate_rampup_kimg=0,
                      tick_kimg_base=60, # note we only use 1/10 of the official implementation. My GPU is too slow...
                      tick_kimg_dict={4: 60, 8:40, 16:20, 32:20, 64:20, 128:20, 256:20, 512:20, 1024:20}
                      ):
    # dnnlib comes with EasyDict
    s = EasyDict()
    s.kimg = cur_nimg  / 1000.0

    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    s.lod = resolution_log2
    s.lod -= np.floor(np.log2(lod_initial_resolution))
    s.lod -= phase_idx
    if lod_transition_kimg > 0:
        s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (resolution_log2 - int(np.floor(s.lod)))
    s.resolution_log2 = int(np.log2(s.resolution))
    s.alpha = 1 - (s.lod - (resolution_log2 - s.resolution_log2))
    assert 0 <= s.alpha <= 1.0

    # Minibatch size.
    s.minibatch = minibatch_dict.get(s.resolution, minibatch_base)
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s


def train_loop(generator,
               discriminator,
               g_optimizer,
               d_optimizer,
               g_loss,
               d_loss,
               initial_resolution,
               data_root,
               minibatch_dict,
               max_minibatch_per_gpu,
               G_lrate_dict,
               G_lrate_base,
               D_lrate_dict,
               D_lrate_base,
               target_resolution_log2=10,
               num_gpus=1,
               total_kimg=15000,  # Total length of the training, measured in thousands of real images
               image_snapshot_ticks=2, # How often to export images
               device='cuda:0',
               cur_nimg=0,
               cur_tick=0,
               prev_resolution=0,
               output_dir='./checks/fake_imgs',
               checkpoint_dir='./checkpoints/',
               ):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # don't do 1

    tick_start_nimg = cur_nimg

    is_resume_training = False
    if cur_nimg != 0:
        is_resume_training = True
        resumed_tick = cur_tick
    else:
        # to always make resumed_tick != current_tick
        resumed_tick = -1

    # for resuming, we don't count previous training's time, which can be fixed
    total_time = .0
    if cur_nimg != 0:
        print('resuming training from tick %d' % cur_tick)
    while cur_nimg < total_kimg * 1000:
        start_time = time.time()
        shed = training_schedule(cur_nimg,
                                 resolution_log2=target_resolution_log2,
                                 num_gpus=num_gpus,
                                 minibatch_dict=minibatch_dict,
                                 max_minibatch_per_gpu=max_minibatch_per_gpu,
                                 lod_initial_resolution=initial_resolution,
                                 G_lrate_dict=G_lrate_dict,
                                 G_lrate_base=G_lrate_base,
                                 D_lrate_dict=D_lrate_dict,
                                 D_lrate_base=D_lrate_base,
                                 )
        if prev_resolution != shed.resolution or is_resume_training:
            # need new size - need a better way
            dataset = dset.ImageFolder(root=data_root,
                                       transform=transforms.Compose([
                                                 transforms.Resize(shed.resolution),
                                                 transforms.CenterCrop(shed.resolution),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
            is_resume_training = False

            # make sure shuffle is True
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=shed.minibatch, shuffle=True, num_workers=4)
            # update learning rate
            g_optimizer.lr = shed.G_lrate
            d_optimizer.lr = shed.D_lrate

        prev_resolution = shed.resolution

        for i, data in enumerate(dataloader, 0):
            # update discriminator
            discriminator.zero_grad()
            set_grad_flag(discriminator, True)
            set_grad_flag(generator, False)
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            # forward pass real batch through discriminator
            real_results = discriminator(real_data, shed.resolution_log2, shed.alpha)
            real_predict = g_loss(real_results, status=True)
            real_predict.backward(retain_graph=True)

            d_loss(real_data, discriminator, shed.resolution_log2, shed.alpha, 5.0)


            # train with all-fake batch
            latents_1 = torch.randn(b_size, 512, device=device)
            fake_data_1 = generator(latents_1, shed.resolution_log2, shed.alpha)
            # use detach so the generator is not updated
            fake_results_1 = discriminator(fake_data_1, shed.resolution_log2, shed.alpha)
           # if status:
           #     return F.softplus(-input[:, 0]).mean()
           # return F.softplus(input[:, 0]).mean()
            loss_fake_d = g_loss(fake_results_1, status=False)
            loss_fake_d.backward()

            loss_d = real_predict + loss_fake_d
            d_optimizer.step()

            # update generator
            generator.zero_grad()

            latents_2 = torch.randn(b_size, 512, device=device)

            set_grad_flag(discriminator, False)
            set_grad_flag(generator, True)

            # another forward pass
            fake_data_2 = generator(latents_2, shed.resolution_log2, shed.alpha)

            fake_results_2 = discriminator(fake_data_2, shed.resolution_log2, shed.alpha)
            # A trick used: assume the labels are correct to reverse the sign in the loss
            loss_fake_g = g_loss(fake_results_2, status=True)
            loss_fake_g.backward()
            g_optimizer.step()

            cur_nimg += batch_size
            # maintenance tasks once per tick
            done = (cur_nimg >= total_kimg * 1000)
            if cur_nimg >= tick_start_nimg + shed.tick_kimg * 1000 or done:
                elapsed_time = time.time() - start_time
                cur_tick += 1
                total_time += elapsed_time
                tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
                tick_start_nimg = cur_nimg
                print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12f sec/tick %-7.1f sec/kimg %-7.2f' % (
                    cur_tick, cur_nimg / 1000.0, shed.lod, shed.minibatch, total_time, elapsed_time, elapsed_time / tick_kimg
                ))
                # for debugging
                print('resolution: %d, alpha: %.3f' % (shed.resolution, shed.alpha))

                print('loss_d %.6f, loss_g %.6f' % (loss_d.item(), loss_fake_g.item()))
                if (cur_tick % image_snapshot_ticks == 0 or done) and (resumed_tick != cur_tick):
                    fixed_noise = torch.randn(10, 512, device=device)
                    with torch.no_grad():
                        fakes = generator(fixed_noise, shed.resolution_log2, shed.alpha)
                    print('saving fake images and checkpoints at tick %d' % cur_tick)
                    vutils.save_image(fakes,
                                      os.path.join(output_dir, 'sample_{}.png'.format(cur_tick)),
                                      padding=2, nrow=5, normalize=True)
                    torch.save(generator.state_dict(),
                               os.path.join(checkpoint_dir,
                                            'generator.%dx%d.%.6f.%d.%d.pt' % (shed.resolution,
                                                                              shed.resolution,
                                                                              shed.alpha,
                                                                              cur_nimg,
                                                                              cur_tick
                                                                           )))
                    torch.save(discriminator.state_dict(),
                               os.path.join(checkpoint_dir,
                                            'discriminator.%dx%d.%.6f.%d.%d.pt' % (shed.resolution,
                                                                                  shed.resolution,
                                                                                  shed.alpha,
                                                                                  cur_nimg,
                                                                                  cur_tick
                                                                           )))


if __name__ == '__main__':
    ############# cmd line parameters ##########
    data_root = args.data_root
    resume_training = args.resume
    if resume_training:
        g_checkpoint_path = args.g_checkpoint
        d_checkpoint_path = args.d_checkpoint

    resolution = args.target_resolution
    n_gpu = args.n_gpu
    ############################################

    # other configs
    batch_size = 16
    num_workers = 4
    # num_epochs = 1000
    g_lr_base = 0.001
    g_lr_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    d_lr_base = 0.001
    d_lr_dict = g_lr_dict
    # 1 GPU setting
    # NOTE for multi-gpu settings, please check official TF implementation
    minibatch_base = 4
    minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
    g_opt_dict = {'beta1': .0, 'beta2': 0.99, 'eplilon': 1e-8}
    d_opt_dict = g_opt_dict
    r1_gamma = 10.0
    initial_resolution = 8

    tick_kimg_base = 160
    tick_kimg_dict = {4: 160, 8: 140, 16: 120, 32: 100, 64: 80, 128: 60, 256: 40, 512: 30, 1024: 20}

    logistic_grad_weight = r1_gamma / 2.0

    device = 'cuda:0' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu'

    with mlflow.start_run():

        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        print('Construting networks...')

        generator = Generator(resolution=resolution)
        generator.to(device)

        discriminator = BasicDiscriminator(resolution=resolution)
        discriminator.to(device)

        # multi-gpu support
        if device == 'cuda:0' and n_gpu > 1:
            generator = nn.DataParallel(generator, list(range(n_gpu)))
            discriminator = nn.DataParallel(discriminator, list(range(n_gpu)))

        d_loss = logisticGradientPenalty
        g_loss = Logistic(device=device).getCriterion

        g_optimizer = optim.Adam(generator.parameters(),
                                 betas=(g_opt_dict['beta1'],
                                 g_opt_dict['beta2']),
                                 eps=g_opt_dict['eplilon'])

        d_optimizer = optim.Adam(discriminator.parameters(),
                                 betas=(d_opt_dict['beta1'],
                                        d_opt_dict['beta2']),
                                 eps=d_opt_dict['eplilon'])


        # these are for resuming training
        cur_nimg = 0
        cur_tick = 0
        prev_resolution = 0
        print('starting training loop...')
        if resume_training:
            try:
                g_info, d_info = get_resume_info_from_checkpoint(g_checkpoint_path, d_checkpoint_path)
                generator.load_state_dict(torch.load(g_checkpoint_path))
                discriminator.load_state_dict(torch.load(d_checkpoint_path))
                cur_nimg = g_info['cur_nimg']
                cur_tick = g_info['cur_tick']
                prev_resolution = g_info['resolution']
            except:
                print('Resume training failed!!!')
                cur_nimg = 0
                cur_tick = 0
                prev_resolution = -1

        train_loop(generator,
                   discriminator,
                   g_optimizer,
                   d_optimizer,
                   g_loss=g_loss,
                   d_loss=d_loss,
                   initial_resolution=initial_resolution,
                   data_root=data_root,
                   minibatch_dict=minibatch_dict,
                   # this is empty for 1 gpu
                   # for multi-gpu, look at the official TF implementation
                   max_minibatch_per_gpu={},
                   G_lrate_dict=g_lr_dict,
                   G_lrate_base=g_lr_base,
                   D_lrate_dict=d_lr_dict,
                   D_lrate_base=d_lr_base,
                   target_resolution_log2=int(np.sqrt(resolution)),       # only gtx 1070, even using 64 is slow
                   num_gpus=1,
                   total_kimg=12000,
                   image_snapshot_ticks=1,
                   device=device,
                   cur_nimg=cur_nimg,
                   cur_tick=cur_tick,
                   prev_resolution=prev_resolution
                   )


