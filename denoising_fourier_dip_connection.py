from __future__ import print_function

import glob
import random
import time

from models import *
from utils.denoising_utils import *
from utils.wandb_utils import *
from utils.freq_utils import *
from utils.common_utils import compare_psnr_y

import torch.optim
import torch.nn.functional as F
from torch.fft import fft2, fft, fftshift, ifft
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbs
import pandas as pd

import os
import wandb
import argparse
import numpy as np
# from torch_dct import dct_2d, idct_2d
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# Fix seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0')
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--input_index', default=1, type=int)
parser.add_argument('--dataset_index', default=0, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
parser.add_argument('--freq_lim', default=8, type=int)
parser.add_argument('--freq_th', default=20, type=int)
parser.add_argument('--noise_depth', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma / 255.

if args.index == -1:
    fnames = sorted(glob.glob('data/denoising_dataset/*.*'))
    fnames_list = fnames
    if args.dataset_index != -1:
        fnames_list = fnames[args.dataset_index:args.dataset_index + 1]
elif args.index == -2:
    base_path = './data/videos/blackswan_cropped_30'
    save_dir = 'plots/{}/denoising'.format(base_path.split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)
    # fnames = sorted(glob.glob('./data/videos/rollerblade/*.png'))
    # fnames = sorted(glob.glob('./data/videos/blackswan/*.png'))
    # fnames = sorted(glob.glob('./data/videos/judo/*.jpg'))
    fnames = sorted(glob.glob(base_path + '/*.jpg'))
    # fnames = sorted(glob.glob('./data/videos/tennis/*.png'))
    fnames_list = fnames
    # fnames_list = np.random.choice(fnames, 8, replace=False)
else:
    fnames = ['data/denoising/F16_GT.png', 'data/inpainting/kate.png', 'data/inpainting/vase.png',
              'data/sr/zebra_GT.png', 'data/denoising/synthetic_img.png', 'data/denoising/synthetic3_img_600.png',
              'data/denoising/synthetic4_img_600.png']
    fnames_list = [fnames[args.index]]

training_times = []
for fname in fnames_list:
    if fname == 'data/denoising/snail.jpg':
        img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_noisy_np = pil_to_np(img_noisy_pil)

        # As we don't have ground truth
        img_pil = img_noisy_pil
        img_np = img_noisy_np

        if PLOT:
            plot_image_grid([img_np], 4, 5)

    elif fname in fnames:
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_norm_np = img_np - img_np.mean()
        output_depth = img_np.shape[0]
        # if args.index == -2:
        #     from utils.video_utils import crop_and_resize
        #     img_np = crop_and_resize(img_np.transpose(1, 2, 0), (192, 384))
        #     img_np = img_np.transpose(2, 0, 1)
        #     img_pil = np_to_pil(img_np)

        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        img_noisy_norm_np = img_noisy_np - img_noisy_np.mean()
        # img_noisy_pil, img_noisy_np = img_pil, img_np

        # if PLOT:
        #     plot_image_grid([img_np, img_noisy_np], 4, 6)
    else:
        assert False

    INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
    pad = 'reflection'
    if INPUT == 'infer_freqs':
        OPT_OVER = 'net,input'
    else:
        OPT_OVER = 'net'
        # OPT_OVER = 'net,input'

    train_input = True if ',' in OPT_OVER else False
    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    LR = args.learning_rate

    OPTIMIZER = 'adam'  # 'LBFGS'
    show_every = 100
    exp_weight = 0.99

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    # img_noisy_norm_torch = img_noisy_torch - img_noisy_torch.mean()
    img_noisy_norm_torch = np_to_torch(img_noisy_norm_np)
    # img_noisy_norm_torch_dct = dct_2d(img_noisy_norm_torch, norm='ortho').type(dtype)

    if fname == 'data/denoising/snail.jpg':
        num_iter = 2400
        input_depth = 3
        figsize = 5

        net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        net = net.type(dtype)

    elif fname in fnames:
        adapt_lim = args.freq_lim

        num_iter = 1800
        figsize = 4
        freq_dict = {
            'method': 'random',
            'cosine_only': False,
            'n_freqs': args.num_freqs,
            'base': 2 ** (adapt_lim / (args.num_freqs - 1)),
        }

        if INPUT == 'noise':
            input_depth = args.noise_depth
        elif INPUT == 'meshgrid':
            input_depth = 2
        else:
            input_depth = args.num_freqs * 4

        net = get_net(input_depth, 'skip', pad, n_channels=output_depth,
                      skip_n33d=128,
                      skip_n33u=128,
                      skip_n11=4,
                      num_scales=5,
                      upsample_mode='bilinear').type(dtype)

        filename = os.path.basename(fname).split('.')[0]
        run = wandb.init(project="Fourier features DIP",
                         entity="impliciteam",
                         tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename,
                               'denoising', '1D'],
                         name='{}_depth_{}_1D'.format(filename, input_depth),
                         job_type='eval',
                         group='Fourier-DIP',
                         mode='offline',
                         save_code=True,
                         notes='Connection between Fourier and DIP / PIP'
                         )

        wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
        # wandb.watch(net, 'all')
        # log_input_images(img_noisy_np, img_np)

        dip_net = skip(
            num_input_channels=input_depth,
            num_output_channels=output_depth,
            num_channels_up=[128] * 5,
            num_channels_down=[128] * 5,
            num_channels_skip=[4] * 5,
            filter_skip_size=1,
            filter_size_up=3,
            filter_size_down=3,
            upsample_mode='bilinear',
            act_fun='LeakyReLU', need_sigmoid=True, need_bias=True, pad=pad
        ).type(dtype)
        dip_net.load_state_dict(torch.load('dip_unet.pth'))

        pip_net = skip(
            num_input_channels=input_depth,
            num_output_channels=output_depth,
            num_channels_up=[128] * 5,
            num_channels_down=[128] * 5,
            num_channels_skip=[4] * 5,
            filter_skip_size=1,
            filter_size_up=1,
            filter_size_down=1,
            upsample_mode='bilinear',
            act_fun='LeakyReLU', need_sigmoid=True, need_bias=True, pad=pad
        ).type(dtype)
        pip_net.load_state_dict(torch.load('pip_unet.pth'))

        conv_layers_dip = [module for module in dip_net.modules() if isinstance(module, nn.Conv2d)]
        conv_layers_pip = [module for module in pip_net.modules() if isinstance(module, nn.Conv2d)]

        net_input_dip = torch.load('noise_input.pt').detach().clone()
        net_input_pip = get_input(input_depth, 'fourier',
                                  (img_pil.size[1], img_pil.size[0]), freq_dict={
                'method': 'log',
                'cosine_only': False,
                'n_freqs': args.num_freqs,
                'base': 2 ** (adapt_lim / (args.num_freqs - 1)),
            }).detach().clone().type(dtype)

        out_dip_np = torch_to_np(dip_net(net_input_dip))
        out_pip_np = torch_to_np(pip_net(net_input_pip))

        print('DIP: {}'.format(compare_psnr(img_np, out_dip_np)))
        print('PIP: {}'.format(compare_psnr(img_np, out_pip_np)))

        dip_convs = [module for module in dip_net.modules() if isinstance(module, nn.Conv2d)]
        pip_convs = [module for module in pip_net.modules() if isinstance(module, nn.Conv2d)]
        net_init_convs = [module for module in net.modules() if isinstance(module, nn.Conv2d)]

        pip_convs[0].weight.data
        # Visualize pip convs
        for layer_n in range(2):
            for p in range(pip_convs[layer_n].weight.data.shape[0]):
                fig = plt.figure()
                plt.plot(pip_convs[1].weight.data[p, :, 0, 0].detach().cpu())
                [plt.axvline(i, ymin=-1, ymax=1) for i in range(0, 31, 8)]
                plt.savefig('pip_conv_{}_{}.png'.format(layer_n, p))
                plt.close(fig)

        dip_first_conv_output = dip_convs[1](net_input_dip)
        pip_first_conv_output = pip_convs[1](net_input_pip)
        dip_first_conv_output_ft = fft2(dip_first_conv_output.detach().cpu(), norm='ortho').abs()

        dip_first_conv_output_np = dip_first_conv_output.detach().cpu().numpy()
        pip_first_conv_output_np = pip_first_conv_output.detach().cpu().numpy()

        kernel_zero_pad = F.pad(dip_convs[1].cpu().weight, (0, 512 - 3, 0, 512 - 3))
        kernel_2d_zero_pad_ft = fft2(kernel_zero_pad.detach(), norm='ortho').abs().cpu()

        net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)
        net_input_ft = fft2(net_input.detach().cpu(), norm='ortho').abs()

        init_conv_pad = F.pad(net_init_convs[1].cpu().weight, (0, 512 - 3, 0, 512 - 3))
        init_conv_pad_ft = fft2(init_conv_pad.detach(), norm='ortho').abs().cpu()

        # 1D
        net_input_1d = get_input(input_depth, INPUT, (img_pil.size[1], 1), freq_dict=freq_dict).squeeze(-1)
        net_input_1d_ft = fft(net_input_1d, norm='ortho').cpu()
        net_input_1d_ft_mag = net_input_1d_ft.abs()

        conv1d = nn.Conv1d(32, 1, kernel_size=3, stride=1)
        kernel_zero_pad_ft = fft(F.pad(conv1d.weight.detach(), (0, 512 - 3)), norm='ortho').cpu()
        kernel_zero_pad_ft_mag = kernel_zero_pad_ft.abs()

        out_1d_ft = net_input_1d_ft * kernel_zero_pad_ft
        out_1d_ft_mag = out_1d_ft.abs()
        out_1d_spatial = ifft(out_1d_ft, norm='ortho').cpu().abs()

        k = 100
        cos_vec = torch.cos((2 * torch.pi / net_input_1d.shape[-1]) * torch.arange(net_input_1d.shape[-1]) * k)
        res = (cos_vec * net_input_1d_ft * kernel_zero_pad_ft).abs()
        # hist_2d, x_edges, y_edges = np.histogram2d(dip_first_conv_output_np[0, :, 5:-5, 5:-5].ravel(),
        #                                            pip_first_conv_output_np[0, :, 5:-6, 5:-6].ravel(),
        #                                            bins=50)
        # # hist_2d_log = np.zeros(hist_2d.shape)
        # # non_zeros = hist_2d != 0
        # # hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        # # plt.imshow(hist_2d_log.T, origin='lower', cmap='gray')
        # # plt.xlabel('DIP signal bin')
        # # plt.ylabel('PIP signal bin')
        # # plt.colorbar()
        # # plt.show()
        # print(mutual_information(hist_2d))

    else:
        assert False

    # net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)
    # Add Projection layer
    # projection = nn.Conv2d(net_input.shape[1], input_depth, kernel_size=(1, 1), stride=1).type(dtype)
    # net = nn.Sequential(
    #     projection,
    #     net
    # )
    # Compute number of parameters
    # s = sum([np.prod(list(p.size())) for p in net.parameters()])
    # print('Number of params: %d' % s)
    # print(net_input)

    observations = []
    for _ in range(1000):
        net_input_1d = get_input(1, INPUT, (img_pil.size[1], 1), freq_dict=freq_dict).view(1, -1)
        net_input_1d_ft = fft(net_input_1d, norm='ortho').cpu()
        observations.append(net_input_1d_ft.abs())

    observations_tensor = torch.cat(observations, dim=0)
    std, mean = torch.std_mean(observations_tensor, unbiased=False)
    jump_val = 8
    indices_vec = torch.arange(0, observations_tensor.shape[1], jump_val)
    observations_tensor_every_n = observations_tensor[:, indices_vec]
    df = pd.DataFrame(observations_tensor_every_n.cpu().numpy(), columns=[str(i.item()) for i in indices_vec])
    fig = plt.figure(figsize=(30, 5))
    sbs.violinplot(data=df)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

    # tables = []
    # for k in range(20):
    #     i = np.random.choice(range(init_conv_pad_ft.shape[0]), 1, replace=False)[0]
    #     j = np.random.choice(range(init_conv_pad_ft.shape[1]), 1, replace=False)[0]
    #
    #     curr_random_init_kernel = init_conv_pad_ft[i, j].cpu()
    #     curr_learned_kernel = kernel_2d_zero_pad_ft[i, j].cpu()
    #
    #     fig, axes = plt.subplots(1, 2)
    #     im1 = axes[0].imshow(curr_random_init_kernel)
    #     axes[0].axis('off')
    #     divider = make_axes_locatable(axes[0])
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im1, cax=cax, orientation='vertical')
    #     axes[0].set_title('Randon - Init ({}, {})'.format(i, j))
    #     im2 = axes[1].imshow(curr_learned_kernel)
    #     axes[1].axis('off')
    #     divider = make_axes_locatable(axes[1])
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im2, cax=cax, orientation='vertical')
    #     axes[1].set_title('Learned ({}, {})'.format(i, j))
    #
    #     wandb.log({'Kernel': fig})
    #     plt.close(fig)
    #
    #
    # #
    # # commit = False
    # # for k in range(dip_first_conv_output_ft.shape[1]):
    # #     if k == dip_first_conv_output_ft.shape[1] - 1:
    # #         commit = True
    # #
    # #     fig1 = plt.figure()
    # #     plt.imshow(dip_first_conv_output_ft[0, k].cpu())
    # #     plt.colorbar()
    # #     plt.title('plane #{}'.format(k))
    # #     plt.axis('off')
    # #     wandb.log({"|FT(out-l1)|": fig1})
    # #     plt.close(fig1)
    # # 1D example
    # tables = []
    # keys = []
    # for i in range(net_input_1d_ft_mag.shape[1]):
    #     data_y = [y.item() for y in net_input_1d_ft_mag[0, i]]
    #     tables.append(data_y)
    #     keys.append(str(i))
    #
    # wandb.log({"|FT(Input)|": wandb.plot.line_series(
    #     xs=np.arange(0, net_input_1d_ft_mag.shape[-1]),
    #     ys=tables,
    #     keys=keys,
    #     title="|FT(Input)|",
    #     xname="samples")})
    #
    #
    # tables = []
    # keys = []
    # for i in range(kernel_zero_pad_ft_mag.shape[1]):
    #     data = [[x, y.item()] for (x, y) in zip(np.arange(0, net_input_1d_ft_mag.shape[-1]), net_input_1d_ft_mag[0, i])]
    #     # data_y = [y.item() for y in kernel_zero_pad_ft_mag[0, i]]
    #     # tables.append(data_y)
    #     # keys.append(str(i))
    #     tables.append(wandb.Table(data=data, columns=["x", "y"]))
    #     wandb.log({"|FT(Input)| - {}".format(i): wandb.plot.line(tables[-1], "x", "y".format(i),
    #                title="|FT(Input)| #{}".format(i))})
    #
    # # wandb.log({"|FT(kernel(zero-pd))|": wandb.plot.line_series(
    # #     xs=np.arange(0, kernel_zero_pad_ft_mag.shape[-1]),
    # #     ys=tables,
    # #     keys=keys,
    # #     title="|FT(kernel(zero-pd))|",
    # #     xname="samples")})
    # tables = []
    # for i in range(kernel_zero_pad_ft_mag.shape[1]):
    #     data = [[x, y.item()] for (x, y) in zip(np.arange(0, kernel_zero_pad_ft_mag.shape[-1]),
    #                                             kernel_zero_pad_ft_mag[0, i])]
    #     tables.append(wandb.Table(data=data, columns=["x", "y"]))
    #     wandb.log({"|FT(Kernel)| - {}".format(i): wandb.plot.line(tables[-1], "x", "y".format(i),
    #                title="|FT(Kernel)| Channel #{}".format(i))})
    #
    # tables = []
    # for i in range(out_1d_ft_mag.shape[1]):
    #     data = [[x, y.item()] for (x, y) in zip(np.arange(0, out_1d_ft_mag.shape[-1]), out_1d_ft_mag[0, i])]
    #     tables.append(wandb.Table(data=data, columns=["x", "y"]))
    #     wandb.log({"|FT(Out)| - {}".format(i): wandb.plot.line(tables[-1], "x", "y".format(i),
    #                title="|FT(Out)| Channel #{}".format(i))})
    #
    # tables = []
    # for i in range(out_1d_spatial.shape[1]):
    #     data = [[x, y.item()] for (x, y) in zip(np.arange(0, out_1d_spatial.shape[-1]), out_1d_spatial[0, i])]
    #     tables.append(wandb.Table(data=data, columns=["x", "y"]))
    #     wandb.log({"|Out| - {}".format(i): wandb.plot.line(tables[-1], "x", "y".format(i),
    #                                                        title="|Out Channel #{}".format(i))})
    #
    # tables = []
    # for i in range(res.shape[1]):
    #     data = [[x, y.item()] for (x, y) in zip(np.arange(0, res.shape[-1]), res[0, i])]
    #     tables.append(wandb.Table(data=data, columns=["x", "y"]))
    #     wandb.log({"|Out - {}| - {}".format(k, i): wandb.plot.line(tables[-1], "x", "y".format(i),
    #                                                        title="|Out ({} row) Channel #{}|".format(k, i))})
