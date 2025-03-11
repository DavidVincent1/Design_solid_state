import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.image import imread
from skimage.feature import peak_local_max

def load_image(path):
    """Loads an image as an numpy array"""
    img = np.array(imread(path), dtype=float)
    return img

SCALE = 42.468E-12 # mètres/pixel


def find_dhkl(image_number, if_plot=True):
    img = load_image(f"Micrographies_par_équipe/Micrographie_{image_number}.tif")
    fft = np.fft.fft2(img)
    fft_shifted = np.abs(np.fft.fftshift(fft))
    peaks = peak_local_max(fft_shifted, min_distance=10, threshold_rel=0.07)  # Détection des pics
    center = np.array(fft_shifted.shape) // 2
    distances_px = np.sqrt((peaks[:, 0] - center[0])**2 + (peaks[:, 1] - center[1])**2) # Distances absolues dans l'espace de Fourier
    if if_plot:
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.imshow(img, origin="lower")
        ax2.imshow(fft_shifted, cmap="inferno")
        ax2.plot(peaks[:,1],peaks[:,0],"o", color="red")
        plt.show()
    sigma_px = 1 # incertitude sur la position des pics
    d_hkl = 1 / distances_px  * SCALE # Conversion en mètres
    sigma_d_hkl = d_hkl*sigma_px/distances_px # Propagation de l'incertitude
    return d_hkl, sigma_d_hkl

for i in range(8,11): # trois images qu'on va utiliser
    d_hkl, sigma_d_hkl = find_dhkl(i)
    for j, d in enumerate(d_hkl):
        print(f"Pic {j+1}: d_hkl = {d:.3e} +/- {sigma_d_hkl[j]:.3e} m")