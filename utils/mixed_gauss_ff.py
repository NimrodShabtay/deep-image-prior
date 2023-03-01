import torch
import torch.nn as nn


class GaussianFourierFeatureTransform(nn.Module):
    """
    Original authors: https://github.com/ndahlquist/pytorch-fourier-feature-networks

    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, num_features*2, width, height].
    """

    def __init__(self, num_channels, num_features=256, scale=10):
        # It generates fourier components of A random frequencies, not all of them.
        # The frequencies are determined by a random normal distribution, multiplied by "scale"
        # So, when "scale" is higher, the fourier features will have higher frequencies
        # In learnable_image_tutorial.ipynb, this translates to higher fidelity images.
        # In other words, 'scale' loosely refers to the X,Y scale of the images
        # With a high scale, you can learn detailed images with simple MLP's
        # If it's too high though, it won't really learn anything but high frequency noise

        super().__init__()

        self.num_channels = num_channels
        self.num_features = num_features

        if isinstance(scale, list):
            scale_ = scale[0]
            temp_scale = scale[1]
        else:
            scale_ = scale
            temp_scale = None
        # freqs are n-dimensional spatial frequencies, where n=num_channels
        ff = torch.abs(torch.randn(num_channels, num_features)) * scale_
        ff = 2 ** ff
        self.freqs = nn.Parameter(ff, requires_grad=False)

        if temp_scale is not None:
            self.freqs[-1, :] = self.freqs[-1, :] * temp_scale/scale_

    def forward(self, x, multiply_only=False):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batch_size, num_channels, height, width = x.shape

        assert num_channels == self.num_channels, \
            "Expected input to have {} channels (got {} channels)".format(self.num_channels, num_channels)

        # Make shape compatible for matmul with freqs.
        # From [B, C, H, W] to [(B*H*W), C].
        x = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_channels)

        # [(B*H*W), C] x [C, F] = [(B*H*W), F]
        x = x.float() @ self.freqs

        # From [(B*H*W), F] to [B, H, W, F]
        x = x.view(batch_size, height, width, self.num_features)
        # From [B, H, W, F] to [B, F, H, W]
        x = x.permute(0, 3, 1, 2)

        x = 2 * torch.pi * x

        if multiply_only:
            output = x
        else:
            output = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
            assert output.shape == (batch_size, 2 * self.num_features, height, width)

        return output
