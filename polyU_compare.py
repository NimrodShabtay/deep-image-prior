import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt
import glob


GT_BASE_DIR = '/home/nimrod/Projects/PolyU-Real-World-Noisy-Images-Dataset/CroppedImages/gt'
DENOISED_BASE_DIR = '/home/nimrod/Projects/KAIR/results/noisy_dncnn_color_blind'

gt_files = sorted(glob.glob(GT_BASE_DIR + '/*.JPG'))
denoised_files = sorted(glob.glob(DENOISED_BASE_DIR + '/*.JPG'))


psnrs = []
for gt, res in zip(gt_files, denoised_files):
    psnrs.append(compare_psnr(plt.imread(gt), plt.imread(res)))


print(np.mean(psnrs))





