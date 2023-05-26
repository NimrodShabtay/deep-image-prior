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
import matplotlib.pyplot as plt

import os
import wandb
import argparse
import numpy as np
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.profiler import profile, record_function, ProfilerActivity
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
parser.add_argument('--input_index', default=0, type=int)
parser.add_argument('--dataset_index', default=0, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
parser.add_argument('--freq_lim', default=8, type=int)
parser.add_argument('--freq_th', default=20, type=int)
parser.add_argument('--sigma', default=25, type=int)
parser.add_argument('--a', default=1., type=float)
parser.add_argument('--supervision', default='gaussian', type=str)
parser.add_argument('--net_type', default='skip', type=str)
parser.add_argument('--num_layers', default=5, type=int)
parser.add_argument('--emb_size', default=128, type=int)
parser.add_argument('--exp_weight', default=0.99, type=float)
parser.add_argument('--reg_noise_std', default=1./30, type=float)
parser.add_argument('--n_iter', default=1801, type=int)
parser.add_argument('--win_len', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
imsize = -1
PLOT = True
sigma = args.sigma
sigma_ = sigma / 255.
gaussian_a = args.a
supervision_type = args.supervision

if args.index == -1:
    fnames = sorted(glob.glob('data/denoising_dataset/*.*'))
    fnames_list = fnames
    if args.dataset_index != -1:
        fnames_list = fnames[args.dataset_index:args.dataset_index + 1]
elif args.index == -2:
    base_path = './data/videos/rollerblade'
    save_dir = 'plots/{}/denoising_pip'.format(base_path.split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)
    fnames = sorted(glob.glob(base_path + '/*.*'))
    fnames_list = fnames
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
        output_depth = img_np.shape[0]

        if supervision_type == 'gaussian':
            img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        elif supervision_type == 'poisson':
            img_noisy_pil, img_noisy_np = get_poisson_image(img_np)
        elif supervision_type == 'fit':
            img_noisy_pil, img_noisy_np = img_pil, img_np
        else:
            raise ValueError('Supervision type not supported {}'.format(supervision_type))

    else:
        assert False

    INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
    pad = 'reflection'
    if INPUT == 'infer_freqs':
        OPT_OVER = 'net,input'
    else:
        OPT_OVER = 'net'

    train_input = True if ',' in OPT_OVER else False
    reg_noise_std = args.reg_noise_std #1. / 30.  # set to 1./20. for sigma=50
    LR = args.learning_rate

    OPTIMIZER = 'adam'  # 'LBFGS'
    show_every = 100
    exp_weight = args.exp_weight

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
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
        # img_f = rfft2(img_noisy_torch, norm='ortho')
        # mag_img_f = torch.abs(img_f).cpu()
        # bins = torch.Tensor([torch.Tensor([0]), *list(2 ** torch.linspace(0, args.freq_lim - 1, args.num_freqs))])
        # hist = torch.histogram(mag_img_f, bins=bins)
        # if hist.hist[-4:].sum() > args.freq_th:
        #     adapt_lim = 8
        # else:
        #     adapt_lim = 7
        adapt_lim = args.freq_lim

        num_iter = args.n_iter
        figsize = 4
        freq_dict = {
            'method': 'log',
            'cosine_only': False,
            'n_freqs': args.num_freqs,
            'base': 2 ** (adapt_lim / (args.num_freqs - 1)),
        }

        if INPUT == 'noise':
            input_depth = 32
        elif INPUT == 'meshgrid':
            input_depth = 2
        else:
            input_depth = args.num_freqs * 4

        ksize = 3
        if args.net_type == 'skip':
            net = get_net(input_depth, 'skip', pad, n_channels=output_depth,
                          skip_n33d=128,
                          skip_n11=4,
                          num_scales=5,
                          act_fun='sin',
                          upsample_mode='bilinear').type(dtype)
        elif args.net_type == 'MLP':
            net = MLP(input_depth, out_dim=output_depth,
                      hidden_list=[args.emb_size for _ in range(args.num_layers)], act='gauss').type(dtype)
                      # n_layers=args.num_layers, n_hidden_units=args.emb_size, act='gauss').type(dtype)
        elif args.net_type == 'FCN':
            net = FCN(input_depth, out_dim=output_depth,
                      hidden_list=[args.emb_size for _ in range(args.num_layers)], ksize=ksize).type(dtype)
        elif args.net_type == 'SimpleFCN':
            net = SimpleFCN(input_depth, out_dim=output_depth,
                            hidden_list=[args.emb_size for _ in range(args.num_layers)], ksize=ksize).type(dtype)
        elif args.net_type == 'SIREN':
            net = SirenConv(in_features=input_depth, hidden_features=args.emb_size, hidden_layers=args.num_layers,
                            out_features=output_depth,
                            outermost_linear=True).type(dtype)
        elif args.net_type == 'FCN_skip':
            net = FCN_skip(input_depth, out_dim=output_depth,
                           hidden_list=[args.emb_size for _ in range(args.num_layers)], ksize=ksize).type(dtype)
        else:
            raise ValueError('net_type {} is not supported'.format(args.net_type))
    else:
        assert False

    net_input = get_input(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]), freq_dict=freq_dict,
                          batch_size=args.batch_size).type(dtype)

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    if train_input:
        net_input_saved = net_input
    else:
        net_input_saved = net_input.detach().clone()

    noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()

    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    psnr_gt_list = []
    psnr_gt_sm_list = []
    i = 0
    input_grads = {idx: [] for idx in range(net_input_saved.shape[0])}
    t_fwd = []
    t_bwd = []
    win_len = args.win_len
    # frames_window = torch.zeros((win_len, 3, img_pil.size[1], img_pil.size[0])).type(dtype)

    def push_to_window(tensor, x):
        return torch.cat((tensor[1:], x), dim=0)


    def closure():
        global i, out_avg, psrn_noisy_last, last_net, net_input, psnr_gt_list, t_fwd, t_bwd, frames_window

        if INPUT == 'noise':
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                net_input = net_input_saved
        elif INPUT == 'fourier':
            net_input = net_input_saved
            # net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
            # net_input_ = net_input_saved + \
            #              (torch.ones_like(net_input_saved).uniform_(-torch.pi, torch.pi) * reg_noise_std)
            # vp_cat = torch.cat((torch.cos(net_input_), torch.sin(net_input_)), dim=-1)
            # net_input = vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)#.type(dtype)

        elif INPUT == 'infer_freqs':
            if reg_noise_std > 0:
                net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                net_input_ = net_input_saved

            net_input = generate_fourier_feature_maps(net_input_, (img_pil.size[1], img_pil.size[0]), dtype)

        else:
            net_input = net_input_saved

        t_s = time.time()

        out = net(net_input)
        # frames_window = push_to_window(frames_window, out.detach())

        t_fwd.append(time.time() - t_s)
        # Smoothing
        if out_avg is None or i <= win_len:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            # out_avg = torch.mean(frames_window, dim=0).unsqueeze(0)


        total_loss = mse(out, img_noisy_torch.repeat(args.batch_size, 1, 1, 1))
        t_s = time.time()
        total_loss.backward()
        t_bwd.append(time.time() - t_s)

        # out_np = out.detach().cpu().numpy()[0]
        # psrn_gt = compare_psnr(img_np, out_np)
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])
        out_np_batch = net(net_input).detach().cpu().numpy()
        out_np = np.mean(out_np_batch, axis=0)
        psrn_noisy = compare_psnr(img_noisy_np, out_np)
        psrn_gt = compare_psnr(img_np, out_np)

        if PLOT and i % show_every == 0:
            print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
                i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))
            wandb.log({'Fitting': wandb.Image(np.clip(np.transpose(out_np, (1, 2, 0)), 0, 1),
                                              caption='step {}'.format(i))}, commit=False)
            # wandb.log({'Fitting-Smooth': wandb.Image(np.clip(np.transpose(out_avg.detach().cpu().numpy()[0],
            #                                                               (1, 2, 0)), 0, 1),
            #                                          caption='step {}'.format(i))}, commit=False)
            # visualize_fourier(out[0].detach().cpu(), iter=i)
            # wandb.log({'mean_inference_psnr_gt': psrn_gt,
            #            **{'inference_{}_psnr_gt'.format(i): compare_psnr(img_np, out_np_batch[i])
            #               for i in range(out_np_batch.shape[0])}}, commit=False)
            wandb.log({'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy, 'psnr_gt_smooth': psrn_gt_sm}, commit=False)
        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -2 and last_net is not None:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss * 0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
                psnr_gt_list.append(psrn_gt)
                psnr_gt_sm_list.append(psrn_gt_sm)

        i += 1

        # Log metrics
        if INPUT == 'infer_freqs':
            visualize_learned_frequencies(net_input_saved)

        wandb.log({'training loss': total_loss.item()}, commit=True)

        if i == num_iter - 2:
            wandb.log({'psnr_gt': psrn_gt, 'psnr_noisy': psrn_noisy, 'psnr_gt_smooth': psrn_gt_sm}, commit=False)
            if args.index == -2:
                print(compare_psnr(img_np, out_np))
                img_final_pil = np_to_pil(np.clip(out_np, 0, 1))
                img_final_pil.save(os.path.join(save_dir, filename + '.png'))
                np.save(os.path.join(save_dir, filename), np.clip(out_np, 0, 1))

        return total_loss


    log_config = {
        "learning_rate": LR,
        "epochs": num_iter,
        'optimizer': OPTIMIZER,
        'loss': type(mse).__name__,
        'input depth': input_depth,
        'input type': INPUT,
        'Train input': train_input,
        'Reg. Noise STD': reg_noise_std,
        'Sigma': sigma
    }
    log_config.update(**freq_dict)
    filename = os.path.basename(fname).split('.')[0]
    run = wandb.init(project="Fourier features DIP",
                     entity="impliciteam",
                     tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, freq_dict['method'],
                           'denoising', supervision_type, args.net_type, str(sigma)],
                     name='{}_depth_{}_{}'.format(filename, input_depth, '{}'.format(INPUT)),
                     job_type='gauss_mlp_{}_{}_{}'.format(args.net_type, INPUT, LR),
                     group='ICCV - Rebuttle',
                     mode='online',
                     save_code=True,
                     config=log_config,
                     notes=''
                     )

    wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
    log_input_images(img_noisy_np, img_np)
    # visualize_fourier(img_noisy_torch[0].detach().cpu(), is_gt=True, iter=0)
    print('Number of params: %d' % s)
    print(net)
    p = get_params(OPT_OVER, net, net_input)
    # if train_input:
    #     if INPUT == 'infer_freqs':
    #         if freq_dict['method'] == 'learn2':
    #             net_input = enc(net_input_saved)
    #         else:
    #             net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
    #                                                       only_cosine=freq_dict['cosine_only'])
    #         log_inputs(net_input)
    #     else:
    #         log_inputs(net_input)

    t = time.time()
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    t_training = time.time() - t
    training_times.append(t_training)
    print('Training time: {}'.format(t_training))
    # wandb.log({'Forward time[sec]': np.mean(t_fwd), 'Backward time[sec]': np.mean(t_bwd),
    #            'Mean_net_training_time': np.mean(t_fwd) + np.mean(t_bwd)})

    # with open(f"results_pip_gt.csv", "a") as f:
    #     f.write(','.join(str(x) for x in psnr_gt_list) + '\n')
    # with open(f"results_pip_gt_smooth.csv", "a") as f:
    #     f.write(','.join(str(x) for x in psnr_gt_sm_list) + '\n')

    if INPUT == 'infer_freqs':
        net_input = generate_fourier_feature_maps(net_input_saved, (img_pil.size[1], img_pil.size[0]), dtype,
                                                  only_cosine=freq_dict['cosine_only'])
        if train_input:
            log_inputs(net_input)
    else:
        net_input = net_input_saved

    # net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
    # net_input_ = net_input_saved + (torch.ones_like(net_input_saved).uniform_(-torch.pi, torch.pi) * reg_noise_std)
    # vp_cat = torch.cat((torch.cos(net_input_), torch.sin(net_input_)), dim=-1)
    # net_input = vp_cat.flatten(-2, -1).permute(0, 3, 1, 2)
    out_np = torch_to_np(net(net_input))
    # out_np_batch = net(net_input).detach().cpu().numpy()
    # out_np = np.mean(out_np_batch, axis=0)
    psrn_gt = compare_psnr(img_np, out_np)

    # wandb.log({'mean_inference_psnr_gt': psrn_gt,
    #            **{'inference_{}_psnr_gt'.format(i): compare_psnr(img_np, out_np_batch[i])
    #               for i in range(out_np_batch.shape[0])}})
    print('avg. training time - {}'.format(np.mean(training_times)))
    log_images(np.array([np.clip(out_np, 0, 1)]), num_iter, task='Denoising')
    wandb.log({'PSNR-Y': compare_psnr_y(img_np, out_np)}, commit=True)
    wandb.log({'PSNR-center': compare_psnr(img_np[:, 5:-5, 5:-5], out_np[:, 5:-5, 5:-5])}, commit=True)

    # if args.index == -2:
    #     print(compare_psnr(img_np, out_np))
    #     wandb.log({'psnr_gt': compare_psnr(img_np, out_np)})
    #     img_final_pil = np_to_pil(np.clip(out_np, 0, 1))
    #     img_final_pil.save(os.path.join(save_dir, filename + '.png'))
    #     np.save(os.path.join(save_dir, filename), np.clip(out_np, 0, 1))

    if False:
        q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
        plt.plot(psnr_gt_list)
        plt.title('max: {}\nlast: {}'.format(max(psnr_gt_list), psnr_gt_list[-1]))
        plt.show()
    run.finish()
