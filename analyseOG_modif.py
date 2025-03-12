import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.image import imread
from skimage.feature import peak_local_max
import matplotlib.patches as patches


def load_image(path):
    """Loads an image as an numpy array"""
    img = np.array(imread(path), dtype=float)
    return img

SCALE = 42.468 # pm/pixel
nb_pix_scalebar = 200
scale_five_pixels = SCALE/nb_pix_scalebar
bar_height = 20  # Hauteur de la scale bar en pixels
image_length = 1024


def find_dhkl(image_number, if_plot=True, thresh=0.01):
    img = load_image(f"Micrographies_par_équipe/Micrographie_{image_number}.tif")
    fft = np.fft.fft2(img)
    fft_shifted = np.abs(np.fft.fftshift(fft))
    peaks = peak_local_max(fft_shifted, min_distance=50, threshold_rel=thresh)  # Détection des pics
    center = np.array(fft_shifted.shape) // 2
    distances_px = np.sqrt((peaks[:, 0] - center[0])**2 + (peaks[:, 1] - center[1])**2) # Distances absolues dans l'espace de Fourier
    fft_magnitude = [np.log(np.abs(i) + 1) for i in fft_shifted]  # log pour un meilleur contraste 
    if if_plot:
        nb_pix_side = 50
        bar_x = 700
        bar_y = fft_magnitude[0].shape[0]-50
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(fft_magnitude, cmap="inferno")
        ax[1].imshow(fft_magnitude, cmap="inferno")
        ax[1].plot(peaks[:,1],peaks[:,0],"o", color="black")
        scale_bar = patches.Rectangle((bar_x, bar_y), nb_pix_scalebar, bar_height, color='white')
        ax[0].add_patch(scale_bar)
        # Ajout du texte de la scale bar
        ax[0].text(bar_x + nb_pix_scalebar / 2, bar_y - 20, f'{scale_five_pixels:.3f} ' + r'$\rm{pm^{-1}}$', 
                    color='white', ha='center', fontsize=14)
        #ax1 = plt.subplot(121)
        #ax2 = plt.subplot(122)
        #ax1.imshow(img, origin="lower")
        #ax2.imshow(fft_shifted, cmap="inferno")
        #ax2.plot(peaks[:,1],peaks[:,0],"o", color="red")
        plt.show()
    sigma_px = 1 # incertitude sur la position des pics
    index_mauvais = []
    #print(distances_px)
    for i in range(len(distances_px)):
        for j in range(len(distances_px)):
            if distances_px[j] != 0 and distances_px[i] > distances_px[j]:
                if np.abs(distances_px[i]-distances_px[j]) > 10:
                    if (distances_px[i] % distances_px[j]) < 10 or abs(distances_px[i] % distances_px[j]-distances_px[j]) < 10:
                        index_mauvais.append(i)
    distances_px_modif = np.delete(distances_px, index_mauvais)
    #print(distances_px_modif)
    d_hkl = 1 / (distances_px_modif/(SCALE * image_length)) # Conversion en mètres
    #print(distances_px)
    sigma_d_hkl = d_hkl*sigma_px/distances_px_modif # Propagation de l'incertitude
    return d_hkl, sigma_d_hkl

def split_d_hkl(d_hkl, sigma_d_hkl, threshold=0.05):
    d_hkl = d_hkl[1:]
    sigma_d_hkl = sigma_d_hkl[1:]
    families = []
    families_std = []
    for i in range(len(d_hkl)):
        if i != 0:
            added = False
            for k in range(len(families)):
                if not added:
                    if np.abs((d_hkl[i]-np.mean(families[k]))/np.mean(families[k])) < threshold:
                        families[k].append(d_hkl[i])
                        families_std[k].append(sigma_d_hkl[i])
                        added = True
                    #elif np.abs(d_hkl[i] % np.mean(families[k])) < 50:
                    #    print("Multiple entier")
                    #    added = True
                    elif k == len(families)-1:
                        families.append([d_hkl[i]])
                        families_std.append([sigma_d_hkl[i]])
                        added = True
        else:
            families.append([d_hkl[i]])
            families_std.append([sigma_d_hkl[i]])
    return families, families_std

for i in range(8,11): # trois images qu'on va utiliser
    if i == 9:
        d_hkl, sigma_d_hkl = find_dhkl(i, thresh=0.005)
        #print(d_hkl)
    elif i == 10:
        d_hkl, sigma_d_hkl = find_dhkl(i, thresh=0.002)
    else:
        d_hkl, sigma_d_hkl = find_dhkl(i, thresh=0.003)
    famille, famille_std = split_d_hkl(d_hkl, sigma_d_hkl)
    for j, d in enumerate(famille):
        print(f"Famille {j+1}: d_hkl = {d[0]:.3e} +/- {famille_std[j][0]:.3e} m")