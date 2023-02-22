import cv2
import numpy as np
import torch.utils.data
import torch.nn as nn
import random
from PIL import Image
from utils.denoising_utils import get_noisy_image, get_poisson_image
from utils.common_utils import np_to_torch, get_input, crop_image, np_to_pil, pil_to_np
from utils.mixed_gauss_ff import GaussianFourierFeatureTransform
from models.downsampler import Downsampler


class DownsamplingSequence:
    def __init__(self, factor=4, output_depth=3, kernel_type='lanczos2'):
        self.factor = factor
        self.output_depth = output_depth
        self.kernel_type = kernel_type
        self.downsampler = Downsampler(n_planes=self.output_depth,
                                       factor=self.factor, kernel_type=self.kernel_type, phase=0.5, preserve_size=True)

    def set_dtype(self, dtype):
        self.downsampler.type(dtype)

    def downsmaple_sequence(self, img_sequence):
        if isinstance(img_sequence, np.ndarray):
            imgs_pil = [np_to_pil(img) for img in img_sequence]
            LR_size = [
                imgs_pil[0].size[0] // self.factor,
                imgs_pil[0].size[1] // self.factor
            ]

            imgs_pil_lr = [img.resize(LR_size, Image.ANTIALIAS) for img in imgs_pil]
            downsampled_seq = [pil_to_np(img_pil) for img_pil in imgs_pil_lr]

        elif isinstance(img_sequence, torch.Tensor):
            self.downsampler.type(img_sequence.dtype)
            downsampled_seq = self.downsampler(img_sequence)

        return downsampled_seq


def crop_and_resize(img, resize):
    """
    Crop and resize img, keeping relative ratio unchanged
    """
    h, w = img.shape[:2]
    source = 1. * h / w
    target = 1. * resize[0] / resize[1]
    if source > target:
        margin = int((h - w * target) // 2)
        img = img[margin:h - margin]
    elif source < target:
        margin = int((w - h / target) // 2)
        img = img[:, margin:w - margin]
    img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
    return img


def load_image(cap, resize=None):
    _, img = cap.read()
    if not resize is None:
        img = crop_and_resize(img, resize).transpose(2, 0, 1)
    else:
        img = np.array(crop_image(Image.fromarray(img), d=64)).transpose(2, 0, 1)
    return img.astype(np.float32) / 255


def select_frames(input_seq, factor=1):
    #Assuming B, C, H, W
    indices = [i for i in range(0, input_seq.shape[0], factor)]
    values = torch.index_select(input_seq, 0,
                              torch.tensor(indices, dtype=torch.int32,
                                           device=input_seq.device))
    return indices, values


class VideoDataset:
    def __init__(self, video_path, input_type, task, crop_shape=None, noise_type='gaussian',
                 sigma=25, mode='random', temp_stride=1, num_freqs=8, batch_size=8, arch_mode='2d',
                 train=True, spatial_factor=4, ff_spatial_scale=6, ff_temporal_scale=2):
        self.sigma = sigma / 255
        self.mode = mode
        cap_video = cv2.VideoCapture(video_path)
        self.n_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.org_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.task = task
        self.downsampler = DownsamplingSequence(factor=spatial_factor) if task == 'spatial_sr' else None
        self.images = []
        self.degraded_images = []
        for fid in range(self.n_frames):
            frame = load_image(cap_video, resize=crop_shape)
            self.images.append(np_to_torch(frame))
            if task == 'denoising':
                if noise_type == 'gaussian':
                    deg_img = np_to_torch(get_noisy_image(frame, self.sigma)[-1])
                elif noise_type == 'poisson':
                    deg_img = np_to_torch(get_poisson_image(frame)[-1])
                else:
                    raise ValueError('noise type {} is not supported'.format(noise_type))

                self.degraded_images.append(deg_img)

            elif task == 'temporal_sr':
                self.degraded_images.append(np_to_torch(frame))
            elif task == 'spatial_sr':
                self.degraded_images.append(np_to_torch(self.downsampler.downsmaple_sequence(np.expand_dims(frame, axis=0))[0]))

        cap_video.release()
        self.images = torch.cat(self.images)
        self.degraded_images = torch.cat(self.degraded_images)
        if crop_shape is None:
            crop_shape = self.images[0].shape[-2:]
        self.crop_height = crop_shape[0]
        self.crop_width = crop_shape[1]
        self.batch_list = None
        self.n_batches = 0
        self.arch_mode = arch_mode
        self.temporal_stride = temp_stride
        self.batch_size = batch_size
        self.input = None
        self.device = 'cuda'
        self.train = train
        self.input_type = input_type
        self.sampled_indices = None
        self.freq_dict = {
            'method': 'mixed',
            'cosine_only': False,
            'n_freqs': num_freqs,
            'base': 2 ** (8 / (num_freqs - 1)),
        }
        self.spatial_x_f = 32
        self.spatial_y_f = 32
        self.spatial_t_f = 2

        # self.input_depth = 32 if input_type == 'noise' else num_freqs * 4
        self.input_depth = num_freqs * 2
        self.ff_spatial_scale = ff_spatial_scale
        self.ff_temporal_scale = ff_temporal_scale

        self.dtype = torch.cuda.FloatTensor

        self.init_batch_list()
        self.init_input()

        if self.train is True:
            if task == 'temporal_sr':
                self.sampled_indices, self.degraded_images_vis = select_frames(self.degraded_images, factor=2)
            else:
                self.sampled_indices = np.arange(0, self.n_frames)

    def get_cropped_video_dims(self):
        return self.crop_height, self.crop_width

    def get_video_dims(self):
        return self.org_height, self.org_width

    def init_batch_list(self, mode=None, temp_stride=None):
        """
        List all the possible batch permutations
        """
        temp_stride = self.temporal_stride if temp_stride is None else temp_stride

        if self.arch_mode == '2d':
            self.batch_list = [(i, temp_stride) for i in range(0, self.n_frames - self.batch_size + 1,
                                                               self.batch_size * temp_stride)]
        else:
            self.batch_list = [(i, self.temporal_stride) for i in range(0, self.n_frames - self.batch_size + 1, 1)]

        self.n_batches = len(self.batch_list)
        if mode is None:
            mode = self.mode

        if mode == 'random':
            random.shuffle(self.batch_list)

    def sample_next_batch(self):
        batch_data = {}
        input_batch, img_degraded_batch, gt_batch = [], [], []

        cur_batch = np.random.choice(self.sampled_indices, self.batch_size)

        for i, fid in enumerate(cur_batch):
            input_batch.append(self.input[fid].unsqueeze(0))
            gt_batch.append(self.images[fid].unsqueeze(0))
            img_degraded_batch.append(self.degraded_images[fid].unsqueeze(0))

        batch_data['cur_batch'] = cur_batch
        batch_data['input_batch'] = torch.cat(input_batch)
        batch_data['img_degraded_batch'] = torch.cat(img_degraded_batch)
        batch_data['gt_batch'] = torch.cat(gt_batch)
        return batch_data

    def next_batch(self):
        if len(self.batch_list) == 0:
            self.init_batch_list()
            return None
        else:
            (batch_idx, batch_stride) = self.batch_list[0]
            self.batch_list = self.batch_list[1:]

            return self.get_batch_data(batch_idx, batch_stride)

    def get_batch_size(self):
        return self.batch_size

    def get_batch_data(self, batch_idx=0, batch_stride=1):
        """
        Collect batch data for certain batch
        """
        if self.arch_mode == '3d':
            cur_batch = range(batch_idx, batch_idx + self.batch_size * batch_stride, batch_stride)
        else:
            if self.mode == 'random':
                cur_batch = np.random.choice([b[0] for b in self.batch_list], self.batch_size, replace=True)
            else:
                cur_batch = range(batch_idx, batch_idx + self.batch_size * batch_stride, batch_stride)

        batch_data = {}
        input_batch, img_degraded_batch, gt_batch = [], [], []

        for i, fid in enumerate(cur_batch):
            input_batch.append(self.input[fid].unsqueeze(0))
            gt_batch.append(self.images[fid].unsqueeze(0))
            img_degraded_batch.append(self.degraded_images[fid].unsqueeze(0))

        batch_data['cur_batch'] = cur_batch
        batch_data['batch_idx'] = batch_idx
        batch_data['batch_stride'] = batch_stride
        batch_data['input_batch'] = torch.cat(input_batch)
        batch_data['img_degraded_batch'] = torch.cat(img_degraded_batch)
        batch_data['gt_batch'] = torch.cat(gt_batch)
        return batch_data

    def add_sequence_positional_encoding(self):
        if self.input_type == 'infer_freqs':
            freqs = self.freq_dict['base'] ** torch.linspace(0., self.freq_dict['n_freqs'] - 6,
                                                             steps=self.freq_dict['n_freqs'])
            self.input = torch.cat([self.input, freqs], dim=0)

    def create_combined_encoding(self):
        from utils.common_utils import get_meshgrid
        spatial_size = (self.crop_height, self.crop_width)
        spatial_feature_extractor = GaussianFourierFeatureTransform(2, self.freq_dict['n_freqs'], self.ff_spatial_scale)
        temporal_feature_extractor = GaussianFourierFeatureTransform(1, self.freq_dict['n_freqs'], self.ff_temporal_scale)
        # Should the amount of frequencies in the spatial and temporal be the same???
        uv_grid_np = get_meshgrid(spatial_size)
        uv_grid_torch = torch.from_numpy(uv_grid_np).unsqueeze(0).repeat(self.n_frames, 1, 1, 1)
        uv_grid = nn.Parameter(uv_grid_torch, requires_grad=False)

        t_grid_np = np.linspace(0, 1, self.n_frames)
        t_grid_torch = torch.from_numpy(t_grid_np).view(-1, 1, 1, 1).repeat(1, 1, *spatial_size)
        t_grid = nn.Parameter(t_grid_torch, requires_grad=False)

        ax_by = nn.Parameter(spatial_feature_extractor(uv_grid, multiply_only=True), requires_grad=False)
        ct = nn.Parameter(temporal_feature_extractor(t_grid, multiply_only=True), requires_grad=False)
        combined_axis_arg = torch.cat([ax_by, ct], dim=1)
        # Should the merge be by concat?

        self.input = torch.cat([torch.sin(combined_axis_arg), torch.cos(combined_axis_arg)], dim=1)

    def init_input(self):
        if self.input_type == 'infer_freqs':
            self.input = self.freq_dict['base'] ** torch.linspace(0.,
                                                                  self.freq_dict['n_freqs'] - 1,
                                                                  steps=self.freq_dict['n_freqs'])
        else:
            self.create_combined_encoding()

        if self.input_type == 'infer_freqs':
            self.input = self.input.unsqueeze(0).repeat(35, 1)

    def generate_random_crops(self, inp, gt):
        B, _, H, W = inp.shape
        Bgt, _, Hgt, Wgt = gt.shape
        Ch, Cw = self.crop_height, self.crop_width
        if Ch == H:
            top = [0] * B
        else:
            top = np.random.randint(0, H - Ch, B)
        if Cw == W:
            left = [0] * B
        else:
            left = np.random.randint(0, W - Cw, B)
        cropped_inp, cropped_gt = [], []
        for i in range(B):
            cropped_inp.append(inp[i, :, top[i]:top[i]+Ch, left[i]:left[i]+Cw])
        for i in range(Bgt):
            cropped_gt.append(gt[i, :, top[i]:top[i] + Ch, left[i]:left[i] + Cw])

        return torch.stack(cropped_inp, dim=0), torch.stack(cropped_gt, dim=0)

    def prepare_batch(self, batch_data):
        if self.train and self.arch_mode == '2d':
            batch_data['input_batch'], batch_data['img_degraded_batch'] = self.generate_random_crops(
                batch_data['input_batch'], batch_data['img_degraded_batch'])

        batch_data['input_batch'] = batch_data['input_batch'].to(self.device)
        batch_data['img_degraded_batch'] = batch_data['img_degraded_batch'].float().to(self.device)

        if self.arch_mode == '3d':
            batch_data['input_batch'] = batch_data['input_batch'].transpose(0, 1).unsqueeze(0)

        return batch_data

    def get_all_inputs(self):
        return self.input

    def get_all_gt(self, numpy=False):
        if numpy:
            ret_val = self.images.detach().cpu().numpy()
        else:
            ret_val = self.images

        return ret_val

    def get_all_degraded(self, numpy=False):
        if self.task == 'temporal_sr':
            degs_imgs = self.degraded_images_vis
        else:
            degs_imgs = self.degraded_images
        if numpy:
            ret_val = degs_imgs.detach().cpu().numpy()
        else:
            ret_val = degs_imgs

        return ret_val
