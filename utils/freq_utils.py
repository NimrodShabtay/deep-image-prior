import torch
from torch.fft import fftshift, rfft2
import pywt
import numpy as np
import matplotlib.pyplot as plt


def analyze_frequencies(img_var, resolution_factor=2):
    import matplotlib.colors as mcolors

    img_f = rfft2(img_var, norm='ortho')
    mag_img_f = torch.abs(img_f).cpu()
    img_shifted_f = fftshift(img_f, dim=[-2])
    mag_shifted_f = torch.abs(img_shifted_f.cpu())
    plt.imshow(mag_shifted_f[0].permute(1, 2, 0))
    plt.colorbar()
    plt.show()
    bins = torch.Tensor([torch.Tensor([0]), * list(2 ** torch.linspace(0, 6, 7))]) / resolution_factor
    hist = torch.histogram(mag_img_f, bins=bins)
    max_freq = hist.bin_edges[int(max(torch.nonzero(hist.hist, as_tuple=True)[0]))]
    print(hist.hist)
    print(bins)
    plt.bar(hist.bin_edges[:-1], hist.hist, width=0.5, color=[k for k in mcolors.BASE_COLORS.keys()])
    plt.ylim([0, 150])
    plt.xlim([0, max_freq])
    plt.show()
    print('max frequency: {}'.format(max_freq.item()))


def visualize_learned_frequencies(learned_frequencies):
    import wandb
    [wandb.log({'learned Frequency #{}'.format(i): learned_frequencies[i]}, commit=False)
     for i in range(learned_frequencies.shape[0])]


def analyze_image(img_torch, size):
    w = pywt.Wavelet('db3')
    size = size  # patch size
    stride = size  # patch stride

    patches = img_torch.unfold(2, size, stride).unfold(3, size, stride).cpu()
    wt_list_cols = []
    for ver_idx in range(patches.shape[2]):
        wt_list_rows = []
        for hor_idx in range(patches.shape[3]):
            current_patch = patches[0, :, ver_idx, hor_idx, :, :]
            wt_list_rows.append(pywt.swt2(current_patch.numpy(), w, level=2))
        wt_list_cols.append(wt_list_rows)
    pass


def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


EPS = np.finfo(float).eps


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    from scipy import ndimage
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) /
              np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) -
              np.sum(s2 * np.log(s2)))

    return mi