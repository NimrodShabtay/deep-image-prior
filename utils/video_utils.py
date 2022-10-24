import cv2
import numpy as np
import torch.utils.data
import random

from utils.denoising_utils import get_noisy_image
from utils.common_utils import np_to_torch, get_input


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
        img = crop_and_resize(img, resize)
    img_convert = img.transpose(2, 0, 1)
    return img_convert.astype(np.float32) / 255


class VideoDataset:
    def __init__(self, video_path, input_type, resize_shape=None, sigma=25, mode='random', temp_stride=1, num_freqs=8,
                 batch_size=8):
        self.sigma = sigma / 255
        self.mode = mode
        cap_video = cv2.VideoCapture(video_path)
        self.n_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.org_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.images = []
        self.noisy_images = []
        for fid in range(self.n_frames):
            frame = load_image(cap_video, resize_shape)
            self.images.append(np_to_torch(frame))
            self.noisy_images.append(np_to_torch(get_noisy_image(frame, self.sigma)[-1]))

        cap_video.release()
        self.images = torch.cat(self.images)
        self.noisy_images = torch.cat(self.noisy_images)

        if resize_shape is not None:
            height = resize_shape[0]
            width = resize_shape[1]
        else:
            height = self.org_height
            width = self.org_width

        self.height = height
        self.width = width
        self.batch_list = None
        self.n_batches = 0
        self.temporal_stride = temp_stride
        self.batch_size = batch_size
        self.input = None
        self.device = 'cuda'
        self.input_type = input_type
        self.freq_dict = {
            'method': 'log',
            'cosine_only': False,
            'n_freqs': num_freqs,
            'base': 2 ** (8 / (num_freqs - 1)),
        }
        self.input_depth = 1 if input_type == 'noise' else num_freqs * 4

        self.init_batch_list()
        self.init_input()

    def get_video_dims(self):
        return self.height, self.width

    def init_batch_list(self):
        """
        List all the possible batch permutations
        """
        self.batch_list = [(i, self.temporal_stride) for i in range(self.n_frames - self.batch_size + 1)]
        self.n_batches = len(self.batch_list)
        if self.mode == 'random':
            random.shuffle(self.batch_list)

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
        cur_batch = range(batch_idx, batch_idx + self.batch_size * batch_stride, batch_stride)
        batch_data = {}
        input_batch, img_noisy_batch, gt_batch = [], [], []

        for i, fid in enumerate(cur_batch):
            input_batch.append(self.input[fid].unsqueeze(0))
            gt_batch.append(self.images[fid].unsqueeze(0))
            img_noisy_batch.append(self.noisy_images[fid].unsqueeze(0))

        batch_data['cur_batch'] = cur_batch
        batch_data['batch_idx'] = batch_idx
        batch_data['batch_stride'] = batch_stride
        batch_data['input_batch'] = torch.cat(input_batch)
        batch_data['img_noisy_batch'] = torch.cat(img_noisy_batch)
        batch_data['gt_batch'] = torch.cat(gt_batch)
        return batch_data

    def add_sequence_positional_encoding(self):
        freqs = self.freq_dict['base'] ** torch.linspace(0., self.freq_dict['n_freqs'],
                                                         steps=self.freq_dict['n_freqs'] * 2)
        spatial_size = self.input.shape[-2:]
        vp = freqs.unsqueeze(1).repeat(1, self.n_frames) * torch.arange(1, self.n_frames+1) / self.n_frames  # FF X frames
        vp = vp.T.view(self.n_frames, self.freq_dict['n_freqs'] * 2, 1, 1).repeat(1, 1, *spatial_size)
        time_pe = torch.cat((torch.cos(vp), torch.sin(vp)), dim=1)
        self.input = torch.cat([self.input, time_pe], dim=1)

    def init_input(self):
        self.input = get_input(self.input_depth, self.input_type, (self.height, self.width),
                               freq_dict=self.freq_dict).repeat(self.n_frames, 1, 1, 1)
        if self.input_type != 'noise':
            self.add_sequence_positional_encoding()

    def prepare_batch(self, batch_data):
        batch_data['input_batch'] = batch_data['input_batch'].to(self.device)
        batch_data['img_noisy_batch'] = batch_data['img_noisy_batch'].to(self.device)

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
        if numpy:
            ret_val = self.noisy_images.detach().cpu().numpy()
        else:
            ret_val = self.noisy_images

        return ret_val