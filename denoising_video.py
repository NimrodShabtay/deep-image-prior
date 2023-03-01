from __future__ import print_function
import random

from models import *
from models.skip_3d import skip_3d, skip_3d_mlp
from utils.denoising_utils import *
from utils.wandb_utils import *
from utils.video_utils import VideoDataset
from utils.common_utils import np_cvt_color
import torch.optim
import matplotlib.pyplot as plt

import os
import wandb
import argparse
import numpy as np
import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from video_consistency_check import SSIM3D

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
parser.add_argument('--input_vid_path', default='', type=str, required=True)
parser.add_argument('--input_index', default=0, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--ff_spatial_scale', default=6, type=int)
parser.add_argument('--ff_temporal_scale', default=2, type=int)
parser.add_argument('--noise_type', default='gaussian', type=str)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = 25
mode = ['2d', '3d'][0]


def eval_video(v_dataset, model, epoch):
    spatial_size = vid_dataset.get_cropped_video_dims()
    img_for_video = np.zeros((v_dataset.n_frames, 3, *spatial_size), dtype=np.uint8)
    img_for_psnr = np.zeros((v_dataset.n_frames, 3, *spatial_size), dtype=np.float32)
    ssim_loss = SSIM3D(window_size=11)

    v_dataset.init_batch_list()
    with torch.no_grad():
        while True:
            batch_data = v_dataset.next_batch()
            if batch_data is None:
                break
            batch_data = v_dataset.prepare_batch(batch_data)

            net_out = model(batch_data['input_batch'])
            if mode == '3d':
                out = net_out.squeeze(0).transpose(0, 1)
            else:
                out = net_out  # N x 3 x H x W

            out_np = out.detach().cpu().numpy()

            img_for_psnr[batch_data['cur_batch']] = out_np
            out_rgb = np.array([np_cvt_color(o) for o in out_np])
            img_for_video[batch_data['cur_batch']] = (out_rgb * 255).astype(np.uint8)

    ignore_start_ind = v_dataset.n_batches * v_dataset.batch_size
    psnr_whole_video = compare_psnr(v_dataset.get_all_gt(numpy=True)[2:ignore_start_ind],
                                    img_for_psnr[2:ignore_start_ind])
    ssim_whole_video = ssim_loss(
        v_dataset.get_all_gt(numpy=False)[2:ignore_start_ind].permute(1, 0, 2, 3).unsqueeze(0),
        torch.from_numpy(img_for_psnr[2:ignore_start_ind]).permute(1, 0, 2, 3).unsqueeze(0))

    wandb.log({'Checkpoint (FPS=10)'.format(epoch): wandb.Video(img_for_video, fps=10, format='mp4'),
               'Checkpoint (FPS=25)'.format(epoch): wandb.Video(img_for_video, fps=25, format='mp4'),
               'Video PSNR': psnr_whole_video,
               'Video 3D-SSIM': ssim_whole_video},
              commit=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'denoising_checkpoint_{}.pth'.format(epoch))


INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
vid_dataset = VideoDataset(args.input_vid_path,
                           input_type=INPUT,
                           num_freqs=args.num_freqs,
                           task='denoising',
                           noise_type=args.noise_type,
                           sigma=sigma,
                           ff_spatial_scale=args.ff_spatial_scale,
                           ff_temporal_scale=args.ff_temporal_scale,
                           crop_shape=None,
                           batch_size=args.batch_size,
                           arch_mode=mode,
                           train=True,
                           temp_stride=1,
                           mode='cont')


# vid_dataset_eval = VideoDataset(args.input_vid_path,
#                                 input_type=INPUT,
#                                 num_freqs=args.num_freqs,
#                                 task='denoising',
#                                 noise_type=args.noise_type,
#                                 crop_shape=None,
#                                 ff_spatial_scale=args.ff_spatial_scale,
#                                 ff_temporal_scale=args.ff_temporal_scale,
#                                 batch_size=args.batch_size,
#                                 arch_mode=mode,
#                                 train=False,
#                                 temp_stride=1,
#                                 mode='cont')
pad = 'reflection'
if INPUT == 'infer_freqs':
    OPT_OVER = 'net,input'
else:
    OPT_OVER = 'net'

train_input = True if ',' in OPT_OVER else False
reg_noise_std = 0  # 1. / 30.  # set to 1./20. for sigma=50
LR = args.learning_rate

OPTIMIZER = 'adam'  # 'LBFGS'
exp_weight = 0.99
if mode == '2d':
    show_every = 300
    n_epochs = 5000


num_iter = 1
figsize = 4

if INPUT == 'noise':
    input_depth = vid_dataset.input_depth
    net = skip_3d(input_depth, 3,
                  num_channels_down=[16, 32, 64, 128, 128, 128],
                  num_channels_up=[16, 32, 64, 128, 128, 128],
                  num_channels_skip=[4, 4, 4, 4, 4, 4],
                  filter_size_up=(3, 3, 3),
                  filter_size_down=(3, 3, 3),
                  filter_size_skip=(1, 1, 1),
                  downsample_mode='stride',
                  need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                  act_fun='LeakyReLU').type(dtype)
else:
    input_depth = args.num_freqs * 2  # 4 * F for spatial encoding, 4 * F for temporal encoding
    if mode == '3d':
        net = skip_3d_mlp(input_depth, 3,
                          num_channels_down=[256, 256, 256, 256, 256, 256],
                          num_channels_up=[256, 256, 256, 256, 256, 256],
                          num_channels_skip=[8, 8, 8, 8, 8, 8],
                          filter_size_up=(1, 1, 1),
                          filter_size_down=(1, 1, 1),
                          filter_size_skip=(1, 1, 1),
                          downsample_mode='stride',
                          need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                          act_fun='LeakyReLU').type(dtype)
    else:
        net = skip(input_depth, 3,
                   num_channels_down=[256, 256, 256, 256, 256, 256],
                   num_channels_up=[256, 256, 256, 256, 256, 256],
                   num_channels_skip=[8, 8, 8, 8, 8, 8],
                   filter_size_up=1,
                   filter_size_down=1,
                   filter_skip_size=1,
                   upsample_mode='bilinear',
                   downsample_mode='stride',
                   need1x1_up=True, need_sigmoid=True, need_bias=True, pad='reflection',
                   act_fun='LeakyReLU').type(dtype)

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

last_net = None
psrn_noisy_last = 0
psnr_gt_list = []
best_psnr_gt = -1.0
best_iter = 0
best_img = None
i = 0


def train_batch(batch_data):
    global j

    net_input_saved = batch_data['input_batch']
    # noise = net_input_saved.detach().clone()
    if INPUT == 'noise':
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            net_input = net_input_saved
    elif INPUT == 'fourier':
        net_input = net_input_saved

    net_out = net(net_input)
    if mode == '3d':
        out = net_out.squeeze(0).transpose(0, 1)  # N x 3 x H x W
    else:
        out = net_out

    total_loss = mse(out, batch_data['img_degraded_batch'])
    total_loss.backward()

    out_np = out.detach().cpu().numpy()
    psrn_noisy = compare_psnr(batch_data['img_degraded_batch'].cpu().numpy(), out_np)
    psrn_gt = compare_psnr(batch_data['gt_batch'].numpy(), out_np)

    wandb.log({'batch loss': total_loss.item(), 'psnr_noisy': psrn_noisy, 'psnr_gt': psrn_gt}, commit=True)
    return total_loss, out_np, psrn_gt


p = get_params(OPT_OVER, net, net_input=vid_dataset.input)
optimizer = torch.optim.Adam(p, lr=LR)
log_config = {
    "learning_rate": LR,
    "iteration per batch": num_iter,
    'Epochs': n_epochs,
    'optimizer': OPTIMIZER,
    'loss': type(mse).__name__,
    'input depth': input_depth,
    'input type': INPUT,
    'Train input': train_input,
    'Reg. Noise STD': reg_noise_std,
    'Sequence length': vid_dataset.batch_size,
    'Video length': vid_dataset.n_frames,
    '# of sequences': vid_dataset.n_batches,
    'noise_type': args.noise_type,
    'Gauss_sigma': sigma,
    'save every': show_every,
    'FF Spatial Frequency Scale': args.ff_spatial_scale,
    'FF Temporal Frequency Scale': args.ff_temporal_scale
}
log_config.update(**vid_dataset.freq_dict)
filename = os.path.basename(args.input_vid_path).split('.')[0]
run = wandb.init(project="Fourier features DIP",
                 entity="impliciteam",
                 tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, vid_dataset.freq_dict['method'],
                       '{}-PIP'.format(mode), args.noise_type],
                 name='{}_depth_{}_{}_{}_sigma_{}'.format(filename, input_depth, '{}'.format(INPUT),
                                                                          mode, sigma),
                 job_type='Combined_FF_{}_{}_{}_{}_{}'.format(INPUT, LR, args.noise_type,
                                                              args.ff_spatial_scale, args.ff_temporal_scale),
                 group='Denoising - Video',
                 mode='online',
                 save_code=True,
                 config=log_config,
                 notes=''
                 )

log_input_video(vid_dataset.get_all_gt(numpy=True),
                vid_dataset.get_all_degraded(numpy=True))

wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
print(net)
n_batches = vid_dataset.n_batches
img_idx = []
for epoch in tqdm.tqdm(range(n_epochs), desc='Epoch', position=0):
    running_psnr = 0.
    running_loss = 0.
    vid_dataset.init_batch_list()
    for batch_cnt in tqdm.tqdm(range(n_batches), desc="Batch", position=1, leave=False):
        batch_data = vid_dataset.next_batch()
        batch_data = vid_dataset.prepare_batch(batch_data)
        for j in range(num_iter):
            optimizer.zero_grad()
            loss, out_sequence, psnr_gt = train_batch(batch_data)
            running_psnr += psnr_gt
            running_loss += loss.item()
            optimizer.step()

    denom = n_batches
    # Log metrics for each epoch
    wandb.log({'epoch loss': running_loss / denom, 'epoch psnr': running_psnr / denom}, commit=False)
    # log_images(np.array([np_cvt_color(o) for o in out_sequence]), epoch, 'Video-Denoising',
    #            commit=False)

    # Infer video:
    if epoch % show_every == 0:
        eval_video(vid_dataset, net, epoch)


# Infer video at the end:
eval_video(vid_dataset, net, epoch)
