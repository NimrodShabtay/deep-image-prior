from __future__ import print_function
from models import *
from utils.wandb_utils import *
import argparse
import os
import random
import glob
import wandb

# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *

# Fix seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--gpu', default='0')
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--input_index', default=1, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--num_freqs', default=8, type=int)
parser.add_argument('--reg_noise_std', default=0.03, type=float)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize = -1
factor = 4
enforse_div32 = 'CROP'  # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True
show_every = 100

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8

if args.index != -1:
    fnames = ['data/sr/zebra_GT.png', 'data/denoising/F16_GT.png', 'data/inpainting/kate.png']
    fnames_list = [fnames[args.index]]
    dataset_tag = 'single_img'
else:
    dataset_path = 'data/sr_datasets/Set5/images'
    fnames_list = sorted(glob.glob(dataset_path + '/*.*'))
    fnames = fnames_list
    dataset_tag = dataset_path.split('/')[-2]

# Starts here
for path_to_image in fnames_list:
    imgs = load_LR_HR_imgs_sr(path_to_image, imsize, factor, enforse_div32)

    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
    output_depth = imgs['LR_np'].shape[0]
    if PLOT:
        # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
        print('PSNR bicubic: %.4f   PSNR nearest: %.4f' % (
            compare_psnr(imgs['HR_np'], imgs['bicubic_np']),
            compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

    INPUT = ['noise', 'fourier', 'meshgrid', 'infer_freqs'][args.input_index]
    pad = 'reflection'
    if INPUT == 'infer_freqs':
        OPT_OVER = 'net,input'
    else:
        OPT_OVER = 'net'

    KERNEL_TYPE = 'lanczos2'
    train_input = True if OPT_OVER != 'net' else False
    LR = args.learning_rate
    tv_weight = 0.0
    OPTIMIZER = 'adam'
    freq_dict = {
        'method': 'log',
        'cosine_only': False,
        'n_freqs': args.num_freqs,
        'base': 2 ** (8 / (args.num_freqs - 1))
    }

    input_depth = args.num_freqs * 4
    if factor == 4:
        num_iter = 2001
        reg_noise_std = args.reg_noise_std
    elif factor == 8:
        num_iter = 4000
        reg_noise_std = 0.05
    else:
        assert False, 'We did not experiment with other factors'

    net_input = get_input(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                          freq_dict=freq_dict).type(dtype)
    print('Input is {}, Depth = {}'.format(INPUT, input_depth))

    NET_TYPE = 'skip'  # UNet, ResNet
    net = get_net(input_depth, 'skip', pad, n_channels=output_depth,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)
    # net = MLP(input_depth, out_dim=3, hidden_list=[256 for _ in range(8)]).type(dtype)
    # net = FCN(input_depth, out_dim=3, hidden_list=[256, 256, 256, 256]).type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)

    img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

    downsampler = Downsampler(n_planes=output_depth,
                              factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)


    def closure():
        global i, net_input, last_net, psnr_LR_last, LR, reduce_lr, best_img, best_psnr_hr, best_iter

        if INPUT == 'noise':
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                net_input = net_input_saved
        elif INPUT == 'fourier':
            net_input = net_input_saved
        elif INPUT == 'infer_freqs':
            if reg_noise_std > 0:
                net_input_ = net_input_saved + (noise.normal_() * reg_noise_std)
            else:
                net_input_ = net_input_saved
            net_input = generate_fourier_feature_maps(net_input_, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                                                      dtype,
                                                      freq_dict['cosine_only'])
        else:
            net_input = net_input_saved

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var)

        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)

        total_loss.backward()

        # Log
        psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
        psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))

        # Backtracking
        if psnr_LR - psnr_LR_last < -5:
            print('Falling back to previous checkpoint.')
            if reduce_lr:
                LR *= 0.1
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            reduce_lr = False
            return total_loss * 0
        else:
            reduce_lr = True
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_LR_last = psnr_LR

        # History
        psnr_history.append([psnr_LR, psnr_HR])

        if i == early_stopping:
            psnr_early_stopping = compare_psnr(put_in_center(np.clip(torch_to_np(out_HR), 0, 1),
                                                             imgs['orig_np'].shape[1:]),
                                               put_in_center(imgs['HR_np'], imgs['orig_np'].shape[1:]))
            log_images(np.array([np.clip(torch_to_np(out_HR), 0, 1), imgs['HR_np']]),
                       early_stopping, task='Out image - {}'.format(early_stopping), psnr=psnr_early_stopping)

        if PLOT and i % show_every == 0:
            print('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR))
            wandb.log({'psnr_hr': psnr_HR, 'psnr_lr': psnr_LR}, commit=False)
            out_HR_np = torch_to_np(out_HR)
            # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)
            if psnr_HR > best_psnr_hr:
                best_psnr_hr = psnr_HR
                best_img = np.copy(out_HR_np)
                best_iter = i
        i += 1

        # Log metrics
        if INPUT == 'infer_freqs':
            visualize_learned_frequencies(net_input_saved)

        wandb.log({'training loss': total_loss.item(), 'Learning Rate': LR}, commit=True)
        return total_loss


    # %%
    log_config = {
        "learning_rate": LR,
        "epochs": num_iter,
        'optimizer': OPTIMIZER,
        'loss': type(mse).__name__,
        'input depth': input_depth,
        'input type': INPUT,
        'Train input': train_input,
        'reg_noise_std': reg_noise_std

    }
    log_config.update(**freq_dict)
    filename = os.path.basename(path_to_image).split('.')[0]
    run = wandb.init(project="Fourier features DIP",
                     entity="impliciteam",
                     tags=['{}'.format(INPUT), 'depth:{}'.format(input_depth), filename, freq_dict['method'],
                           'denoising_dataset', dataset_tag, 'Conv3x3'],
                     name='{}_depth_{}_{}'.format(filename, input_depth, '{}'.format(INPUT)),
                     job_type='{}_{}_{}'.format(dataset_tag, INPUT, LR),
                     group='Super-Resolution - Dataset x{}'.format(factor),
                     mode='online',
                     save_code=True,
                     config=log_config,
                     notes='Baseline'
                     )

    wandb.run.log_code(".", exclude_fn=lambda path: path.find('venv') != -1)
    log_input_images(imgs['LR_np'], imgs['HR_np'])
    psnr_history = []
    if train_input:
        net_input_saved = net_input
    else:
        net_input_saved = net_input.detach().clone()

    noise = torch.rand_like(net_input) if INPUT == 'infer_freqs' else net_input.detach().clone()

    last_net = None
    psnr_LR_last = 0
    reduce_lr = True
    best_psnr_hr = -1.0
    best_img = None
    best_iter = 0
    i = 0
    early_stopping = 2000

    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: %d' % s)
    print(net)

    p = get_params(OPT_OVER, net, net_input)
    if train_input:
        if INPUT == 'infer_freqs':
            net_input = generate_fourier_feature_maps(net_input_saved, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                                                      dtype,
                                                      freq_dict['cosine_only'])
        else:
            log_inputs(net_input)

    optimize(OPTIMIZER, p, closure, LR, num_iter)
    if INPUT == 'infer_freqs':
        net_input = generate_fourier_feature_maps(net_input_saved, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]),
                                                  dtype,
                                                  freq_dict['cosine_only'])
        if train_input:
            log_inputs(net_input)
    else:
        net_input = net_input_saved

    out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
    result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])
    plt.imsave('{}_{}.png'.format(filename, INPUT), result_deep_prior.transpose(1, 2, 0))
    # For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
    plot_image_grid([imgs['HR_np'],
                     imgs['bicubic_np'],
                     out_HR_np], factor=4, nrow=1)

    last_psnr_in_center = compare_psnr(result_deep_prior, put_in_center(imgs['HR_np'], imgs['orig_np'].shape[1:]))
    # log_images(np.array([np.clip(out_HR_np, 0, 1), imgs['HR_np']]), num_iter, task='Super-Resolution',
    #            psnr=last_psnr_in_center)
    # log_images(np.array([np.clip(best_img, 0, 1), imgs['HR_np']]), best_iter, task='Best Image', psnr=best_psnr_hr)
    log_images(np.array([np.clip(out_HR_np, 0, 1)]), num_iter, task='Super-Resolution',
               psnr=last_psnr_in_center)
    log_images(np.array([np.clip(best_img, 0, 1)]), best_iter, task='Best Image', psnr=best_psnr_hr)

    # Calc PSNR-Y for comparison to DIP
    q1 = result_deep_prior[:3].sum(0)
    t1 = np.where(q1.sum(0) > 0)[0]
    t2 = np.where(q1.sum(1) > 0)[0]
    if output_depth > 1:
        psnr_y = compare_psnr_y(imgs['orig_np'][:3, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4],
                                result_deep_prior[:3, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4])
    else:
        psnr_y = compare_psnr(imgs['orig_np'][:1, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4],
                                result_deep_prior[:1, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4])
    wandb.log({'PSNR-Y': psnr_y}, commit=True)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot([h[0] for h in psnr_history])
    axes[0].set_title('LR PSNR\nmax: {:.3f}'.format(max([h[0] for h in psnr_history])))
    axes[1].plot([h[1] for h in psnr_history])
    axes[1].set_title('HR PSNR\nmax: {:.3f}'.format(max([h[1] for h in psnr_history])))
    # plt.savefig('curve_{}.png'.format(num_iter))
    plt.show()
    run.finish()
