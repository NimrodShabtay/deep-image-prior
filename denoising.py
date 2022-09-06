from __future__ import print_function

from models import *
from utils.denoising_utils import *

import torch
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

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--gpu', default='0')
parser.add_argument('--index', default=0)
parser.add_argument('--input_index', default=0)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.
train_input = True

fnames = ['data/denoising/F16_GT.png', 'data/inpainting/kate.png', 'data/inpainting/vase.png', 'data/sr/zebra_GT.png']
fname = fnames[args.index]

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

    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

    # if PLOT:
    #     plot_image_grid([img_np, img_noisy_np], 4, 6)
else:
    assert False


INPUT = ['noise', 'fourier', 'meshgrid'][args.input_index]
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30. if INPUT=='noise' else 0.  # set to 1./20. for sigma=50
LR = 0.01

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
    num_iter = 5000
    input_depth = 32
    figsize = 4

    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

else:
    assert False

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype)

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
if train_input:
    net_input_saved = net_input.requires_grad_(True)

else:
    net_input_saved = net_input.detach().clone()

noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0
psnr_gt_list = []
indices = torch.arange(0, input_depth, dtype=torch.float)
sample_freqs = True if INPUT == 'fourier' else False
i = 0


def log_images(array_of_imgs, iter):
    B, C, H, W  = array_of_imgs.shape
    images_np = np.zeros((H, B*W, C), dtype=np.float32)
    for i in range(B):
        images_np[:, i*W:W*(i+1), :] = np.transpose(array_of_imgs[i], (1, 2, 0))

    wandb.log({'Denoising': wandb.Image(images_np, caption='Iteration #{}'.format(iter))})


def log_inputs(inputs):
    inputs_ = inputs.squeeze(0).detach().cpu().numpy()
    C, H, W = inputs_.shape
    inputs_arr = np.zeros((C, H, W), np.float32)
    for channel_idx in range(C):
        inputs_arr[channel_idx] = inputs_[channel_idx, :, :]

    wandb.log({'Input': [wandb.Image(inputs_arr[ch_idx], caption='channel #{}'.format(ch_idx)) for ch_idx in range(C)]},
              commit=False)


def closure():
    global i, out_avg, psrn_noisy_last, last_net, net_input, psnr_gt_list, indices

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    if sample_freqs:
        if i % 8000 == 0:  # sample freq
            indices = torch.multinomial(torch.arange(0, net_input_saved.size(1), dtype=torch.float),
                                        input_depth, replacement=False)

            assert len(torch.unique(indices)) == input_depth
            print(indices)

        net_input = net_input_saved[:, indices, :, :]

    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    if PLOT and i % show_every == 0:
        print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
            i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))
        psnr_gt_list.append(psrn_gt)
        if train_input:
            log_inputs(net_input)

        wandb.log({'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy}, commit=False)

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    wandb.log({'training loss': total_loss.item()}, commit=True)
    return total_loss


log_config = {
    "learning_rate": LR,
    "epochs": num_iter,
    'optimizer': OPTIMIZER,
    'loss': type(mse).__name__,
    'input depth': input_depth,
    'input type': INPUT,
    'Train input': train_input
}
run = wandb.init(project="Fourier features DIP",
                     entity="impliciteam",
                     tags=['random_ff', 'train_input'],
                     name='{}'.format(os.path.basename(fname).split('.')[0]),
                     job_type='train',
                     group='Denoising',
                     mode='online',
                     save_code=True,
                     config=log_config,
                     notes='Random FF')

wandb.run.log_code(".")
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = torch_to_np(net(net_input))
log_images(np.array([np.clip(out_np, 0, 1), img_np]), num_iter)
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
plt.plot(psnr_gt_list)
plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
plt.show()