import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.video import r3d_18
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import glob
from PIL import Image
from math import exp
import os
from utils.common_utils import crop_image
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class r2p1d18_loss(nn.Module):
    def __init__(self, requires_grad=False, loss_func=torch.nn.SmoothL1Loss(), compute_single_loss=True):
        super().__init__()
        a = r3d_18(pretrained=True)
        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stem = a.stem
        self.slice1 = a.layer1
        self.slice2 = a.layer2
        self.slice3 = a.layer3
        self.slice4 = a.layer4

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.loss_func = loss_func
        self.compute_single_loss = compute_single_loss  # False for vodeometric evaluate

    def forward(self, X1, X2):
        out = []

        for X in [X1, X2]:
            feat_list = []
            # print('passinslices')
            X = self.normalize_batch(X)
            X = self.reshape_batch(X)
            h = self.stem(X)
            feat_list.append(h)

            h = self.slice1(h)
            feat_list.append(h)

            h = self.slice2(h)
            feat_list.append(h)

            if self.compute_single_loss:
                out.append(feat_list)
                continue
            h = self.slice3(h)
            feat_list.append(h)
            h = self.slice4(h)
            feat_list.append(h)
            out.append(feat_list)
        losses = []
        for i in range(len(feat_list)):
            loss = self.loss_func(out[0][i], out[1][i])
            losses.append(loss)
            pass
            # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
            # out.append(vgg_outputs(h_relu1_2,0,0,0))# h_relu2_2, h_relu3_3, h_relu4_3))
        losses = torch.stack(losses)
        if self.compute_single_loss:
            losses = torch.mean(losses)
        # https://github.com/pytorch/examples/blob/d91adc972cef0083231d22bcc75b7aaa30961863/fast_neural_style/neural_style/vgg.py
        return losses

    def normalize_batch(self, batch):
        # normalize using imagenet mean and std
        # mean = batch.new_tensor([0.43216, 0.394666, 0.37645]).view(-1, 1, 1)
        # std = batch.new_tensor([0.22803, 0.22145, 0.216989]).view(-1, 1, 1)
        # batch = batch.div_(255.0)
        T = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        return T(batch)

    def reshape_batch(self, batch):
        # batch : (NS)CHW
        NS, C, H, W = batch.shape
        if not batch.is_contiguous():
            batch = batch.contiguous()
        batch = batch.view(NS // 7, 7, C, H, W)  # NSCHW
        return batch.permute(0, 2, 1, 3, 4)  # NCSHW


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


def avg_psnr(gt, pred):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(gt, pred)


def main():
    from utils.video_utils import VideoDataset, load_image

    dataset = {
        'rollerblade': {
            'gt': './data/eval_vid/clean_videos/rollerblade.avi',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/rollerblade_5000.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/rollerblade.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/rollerblade_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': './data/eval_vid/spatial_sr/pip/rollerblade_pip.mp4',
                'dip': './data/eval_vid/spatial_sr/dip/rollerblade_dip.mp4',
                '3d-dip': './data/eval_vid/spatial_sr/3d_dip/rollerblade_3d_dip_sr.mp4',
            },
            'ignore_index': 5
        },
        'judo': {
            'gt': './data/eval_vid/clean_videos/judo.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/judo.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/judo.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/judo_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': './data/eval_vid/spatial_sr/pip/judo_pip.mp4',
                'dip': './data/eval_vid/spatial_sr/dip/judo_dip.mp4',
                '3d-dip': './data/eval_vid/spatial_sr/3d_dip/judo_3d_dip_sr.mp4',
            },
            'ignore_index': 4
        },
        'dog': {
            'gt': './data/eval_vid/clean_videos/dog.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/dog.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/dog.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/dog_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': './data/eval_vid/spatial_sr/pip/dog_10000.mp4',
                'dip': './data/eval_vid/spatial_sr/dip/dog_dip.mp4',
                '3d-dip': './data/eval_vid/spatial_sr/3d_dip/dog_3d_dip_spatial_sr.mp4'
            },
            'ignore_index': 1
        },
        'camel': {
            'gt': './data/eval_vid/clean_videos/camel_24_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/camel_24_inc_capacity_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/camel_24_frames_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/camel_3d_dip.mp4',
            },
            'spatial_sr': {
                'pip': './data/eval_vid/spatial_sr/pip/camel_24_pip.mp4',
                'dip': './data/eval_vid/spatial_sr/dip/camel_24_frames_dip.mp4',
                '3d-dip': './data/eval_vid/spatial_sr/3d_dip/camel_3d_dip.mp4',
            },
            'ignore_index': 0
        },
        'sheep': {
            'gt': './data/eval_vid/clean_videos/sheep_20_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/sheep_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/sheep_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/sheep_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 2
        },
        'soccerball': {
            'gt': './data/eval_vid/clean_videos/soccerball_20_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/soccerball_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/soccerball_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/soccerball_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 2
        },
        'surf': {
            'gt': './data/eval_vid/clean_videos/surf_21_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/surf_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/surf_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/surf_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 0
        },
        'tractor': {
            'gt': './data/eval_vid/clean_videos/tracktor_sand_21_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/tractor_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/tractor_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/tractor_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 0
        },
        'blackswan': {
            'gt': './data/eval_vid/clean_videos/blackswan_21_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/blackswan_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/blackswan_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/blackswan_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 0
        },
        'car_shadow': {
            'gt': './data/eval_vid/clean_videos/car_shadow_21_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/car_shadow_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/car_shadow_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/car_shadow_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 0
        },
        'train': {
            'gt': './data/eval_vid/clean_videos/train_21_frames.mp4',
            'denoising': {
                'pip': './data/eval_vid/denoised_videos/pip/train_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/train_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/train_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 0
        },
        'bear': {
            'gt': './data/eval_vid/clean_videos/bear_20_frames.mp4',
            'denoising': {
                # 'pip': './data/eval_vid/denoised_videos/pip/sheep_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/bear_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/bear_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 2
        },
        'bike_picking': {
            'gt': './data/eval_vid/clean_videos/bike_picking_20_frames.mp4',
            'denoising': {
                # 'pip': './data/eval_vid/denoised_videos/pip/sheep_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/bike_picking_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/bike_picking_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 2
        },
        'soupbox': {
            'gt': './data/eval_vid/clean_videos/soupbox_20_frames.mp4',
            'denoising': {
                # 'pip': './data/eval_vid/denoised_videos/pip/sheep_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/soapbox_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/soapbox_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 2
        },
        'car_turn': {
            'gt': './data/eval_vid/clean_videos/car_turn_20_frames.mp4',
            'denoising': {
                # 'pip': './data/eval_vid/denoised_videos/pip/sheep_pip.mp4',
                'dip': './data/eval_vid/denoised_videos/frame_by_frame/car_turn_dip.mp4',
                '3d-dip': './data/eval_vid/denoised_videos/3d_dip/car_turn_3d_dip_10.mp4',
            },
            'spatial_sr': {
                'pip': '',
                'dip': '',
                '3d-dip': ''
            },
            'ignore_index': 2
        },
    }
    task = ['denoising', 'spatial_sr'][1]
    names = ['sheep', 'soccerball', 'tractor', 'blackswan', 'car_shadow', 'train', 'surf', 'bear', 'bike_picking',
             'car_turn', 'soupbox', 'camel', 'rollerblade', 'judo', 'dog']
    for name in names:
        print('\n')
        print(name)
        print('-' * 20)
        chosen_video = dataset[name]
        remove_edges_start_index = chosen_video['ignore_index']
        if 0 <= remove_edges_start_index < 2:
            remove_edges_start_index = 2

        dip_ref = []
        gt_ref = []
        pip_ref = []
        pip_binary_ref = []
        dip_binary_ref = []
        dip_3d_ref = []
        d = 64
        try:
            # for img_path in sorted(glob.glob('./plots/{}_20_frames/denoising/*.png'.format(name))):
            #     dip_ref.append(np.array(crop_image(Image.open(img_path), d=64)).transpose(2, 0, 1).astype(np.float32) / 255)
            #
            # dip_ref = torch.from_numpy(np.stack(dip_ref)).cuda()
            # dip_ref = dip_ref[2:-(remove_edges_start_index)]

            for gt_path in sorted(glob.glob('./data/videos/{}_20_frames/*.*'.format(name))):
                gt_ref.append(np.array(crop_image(Image.open(gt_path), d=64)).transpose(2, 0, 1).astype(np.float32) / 255)

            gt_ref = torch.from_numpy(np.stack(gt_ref))#.cuda()
            gt_ref = gt_ref[2:-(remove_edges_start_index)]

            # for img_path in sorted(glob.glob('./plots/{}_20_frames/denoising_pip/*.png'.format(name))):
            #     pip_ref.append(np.array(crop_image(Image.open(img_path), d=64)).transpose(2, 0, 1).astype(np.float32) / 255)
            #
            # pip_ref = torch.from_numpy(np.stack(pip_ref)).cuda()
            # pip_ref = pip_ref[2:-(remove_edges_start_index)]

            for img_path in sorted(glob.glob('./data/eval_vid/spatial_sr/3d_dip/{}/*.*'.format(name))):
                dip_3d_ref.append(
                    np.array(crop_image(Image.open(img_path), d=d)).transpose(2, 0, 1).astype(np.float32) / 255)

            dip_3d_ref = torch.from_numpy(np.stack(dip_3d_ref))#.cuda()
            dip_3d_ref = dip_3d_ref[2:-remove_edges_start_index]

            ssim_loss = SSIM3D(window_size=11)
            print('3D-SSIM')
            # print('dip (frames): {:.4f}'.format(ssim_loss(gt_ref.permute(1, 0, 2, 3).unsqueeze(0),
            #                                               dip_ref.permute(1, 0, 2, 3).unsqueeze(0))))
            # print('pip (frames): {:.4f}'.format(ssim_loss(gt_ref.permute(1, 0, 2, 3).unsqueeze(0),
            #                                               pip_ref.permute(1, 0, 2, 3).unsqueeze(0))))
            print('3d-dip: {:.4f}'.format(ssim_loss(gt_ref.permute(1, 0, 2, 3).unsqueeze(0),
                                                    dip_3d_ref.permute(1, 0, 2, 3).unsqueeze(0))))

            print('Avg. PSNR')
            # print('dip (frames): {:.4f}'.format(avg_psnr(gt_ref.cpu().numpy(), dip_ref.cpu().numpy())))
            # print('pip (frames): {:.4f}'.format(avg_psnr(gt_ref.cpu().numpy(), pip_ref.cpu().numpy())))
            print('3d-dip: {:.4f}'.format(avg_psnr(gt_ref.cpu().numpy(), dip_3d_ref.cpu().numpy())))
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    main()
