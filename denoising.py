from __future__ import print_function

import glob
import random

from models import *
from utils.denoising_utils import *
from utils.wandb_utils import *
from utils.freq_utils import *
from utils.common_utils import compare_psnr_y

import torch.optim
import matplotlib.pyplot as plt

import os
import wandb
import argparse
import numpy as np
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

# Fix seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0')
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--input_index', default=0, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.

if args.index != -1:
    fnames = ['data/denoising/F16_GT.png', 'data/inpainting/kate.png', 'data/inpainting/vase.png', 'data/sr/zebra_GT.png']
    fnames_list = [fnames[args.index]]
else:
    fnames_list = sorted(glob.glob('data/denoising_dataset/*.*'))
    fnames = fnames_list

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
        # Add synthetic noise
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        output_depth = img_np.shape[0]
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

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

    train_input = True if ',' in OPT_OVER else False
    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    LR = args.learning_rate

    OPTIMIZER = 'adam'  # 'LBFGS'
    show_every = 100
    exp_weight = 0.99

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
        num_iter = 1801
        figsize = 4
        freq_dict = {
            'method': 'log',
            'cosine_only': False,
            'n_freqs': args.num_freqs,
            'base': 2 ** (8 / (args.num_freqs-1)),
        }

        input_depth = args.num_freqs * 4
        net = get_net(input_depth, 'skip', pad, n_channels=output_depth,
                      skip_n33d=128,
                      skip_n33u=128,
                      skip_n11=4,
                      num_scales=5,
                      upsample_mode='bilinear').type(dtype)

        # net = MLP(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)
        # net = FCN(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)
    else:
        assert False

    enc = LearnableFourierPositionalEncoding(2, (img_pil.size[1], img_pil.size[0]), 256, 128, input_depth, 10).type(dtype)
    net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict).type(dtype)

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    # analyze_image(img_noisy_torch, size=64)
    if train_input:
        net_input_saved = net_input
    else:
        net_input_saved = net_input.detach().clone()

    noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()

    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    psnr_gt_list = []
    best_psnr_gt = -1.0
    best_iter = 0
    best_img = None
    i = 0
    early_stopping = 1800

    def closure():
        global i, out_avg, psrn_noisy_last, last_net, net_input, psnr_gt_list, best_img, best_psnr_gt, best_iter

        if INPUT == 'noise':
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                net_input = net_input_saved
        elif INPUT == 'fourier':
            # net_input = net_input_saved[:, indices, :, :]
            net_input = net_input_saved
        elif INPUT == 'infer_freqs':
            if reg_noise_std > 0:
                net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                net_input_ = net_input_saved

            if freq_dict['method'] == 'learn2':
                net_input = enc(net_input_)
            else:
                net_input = generate_fourier_feature_maps(net_input_,  (img_pil.size[1], img_pil.size[0]), dtype)
        else:
            net_input = net_input_saved

        out = net(net_input)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()

        out_np = out.detach().cpu().numpy()[0]
        psrn_noisy = compare_psnr(img_noisy_np, out_np)
        psrn_gt = compare_psnr(img_np, out_np)
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

        if i == early_stopping:
            log_images(np.array([np.clip(out_np, 0, 1), img_np]), early_stopping,
                       task='Out image - {}'.format(early_stopping), psnr=psrn_gt)

        if PLOT and i % show_every == 0:
            print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
                i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))
            psnr_gt_list.append(psrn_gt)

            wandb.log({'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy, 'psnr_gt_smooth': psrn_gt_sm}, commit=False)
            if psrn_gt > best_psnr_gt:
                best_psnr_gt = psrn_gt
                best_img = np.copy(out_np)
                best_iter = i
        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -2:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss * 0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy

        i += 1

        # Log metrics
        if INPUT == 'infer_freqs':
            visualize_learned_frequencies(net_input_saved)

        wandb.log({'training loss': total_loss.item()}, commit=True)
        return total_loss

    log_config = {
        "learning_rate": LR,
        "epochs": num_iter,
        'optimizer': OPTIMIZER,
        'loss': type(mse).__name__,
        'input depth': input_depth,
        'input type': INPUT,
        'Train input': train_input,
        'Reg. Noise STD': reg_noise_std
    }
    log_config.update(**freq_dict)
    filename = os.path.basename(fname).split('.')[0]
    run = wandb.init(project="Fourier features DIP",
                     entity="impliciteam",
                     tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, freq_dict['method'],
                           'dataset_comp', 'Conv3x3'],
                     name='{}_depth_{}_{}'.format(filename, input_depth, '{}'.format(INPUT)),
                     job_type='{}_{}'.format(INPUT, LR),
                     group='Denoising - Dataset',
                     mode='online',
                     save_code=True,
                     config=log_config,
                     notes='Baseline'
                     )

    wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
    log_input_images(img_noisy_np, img_np)
    print(net)
    p = get_params(OPT_OVER, net, net_input, input_encoder=enc)
    if train_input:
        if INPUT == 'infer_freqs':
            if freq_dict['method'] == 'learn2':
                net_input = enc(net_input_saved)
            else:
                net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
                                                          only_cosine=freq_dict['cosine_only'])
            log_inputs(net_input)
        else:
            log_inputs(net_input)

    optimize(OPTIMIZER, p, closure, LR, num_iter)
    if INPUT == 'infer_freqs':
        if freq_dict['method'] == 'learn2':
            net_input = enc(net_input_saved)
        else:
            net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
                                                      only_cosine=freq_dict['cosine_only'])
        if train_input:
            log_inputs(net_input)
    else:
        net_input = net_input_saved

    out_np = torch_to_np(net(net_input))
    # log_images(np.array([np.clip(out_np, 0, 1), img_np]), num_iter, task='Denoising')
    # log_images(np.array([np.clip(best_img, 0, 1), img_np]), best_iter, task='Best Image', psnr=best_psnr_gt)
    log_images(np.array([np.clip(out_np, 0, 1)]), num_iter, task='Denoising')
    log_images(np.array([np.clip(best_img, 0, 1)]), best_iter, task='Best Image', psnr=best_psnr_gt)
    wandb.log({'PSNR-Y': compare_psnr_y(img_np, out_np)}, commit=True)
    q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
    plt.plot(psnr_gt_list)
    plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
    plt.show()
    run.finish()
