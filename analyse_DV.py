import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cv2
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches


# Facteur de conversion pixel à m
scale_pm_per_pixel = 42.468
nb_pix_scalebar = 20
scale_five_pixels = scale_pm_per_pixel*nb_pix_scalebar
bar_height = 2  # Hauteur de la scale bar en pixels

# Lire les images
data_8 = tf.imread(f'Micrographies_par_équipe\Micrographie_8.tif')
data_9 = tf.imread(f'Micrographies_par_équipe\Micrographie_9.tif')
data_10 = tf.imread(f'Micrographies_par_équipe\Micrographie_10.tif')
data = [data_8, data_9, data_10]


def find_largest_inscribed_square(data):
    # Masque incluant seulement les pixels non nuls
    mask = data > 0
    y_indices, x_indices = np.where(mask)

    # Zone contennat les pixels non nuls
    points = np.column_stack((x_indices, y_indices))
    hull = ConvexHull(points)

    # Centre de la zone
    center_x, center_y = np.mean(points[hull.vertices], axis=0)

    # Rayon maximal possible (distance du centre aux bords du polygone)
    distances = np.linalg.norm(points[hull.vertices] - [center_x, center_y], axis=1)
    max_radius = np.min(distances)

    # Taille du plus grand carré dans un cercle
    side_length = int(2 * max_radius / np.sqrt(2))

    # Coins du carré
    half_side = side_length // 2
    start_x, end_x = int(center_x - half_side), int(center_x + half_side)
    start_y, end_y = int(center_y - half_side), int(center_y + half_side)

    # Carré final
    cropped_data = data[start_y:end_y, start_x:end_x]

    return cropped_data

# Appliquer la fonction sur les images
cropped_data = [find_largest_inscribed_square(i) for i in data]
fft_data = [np.fft.fft2(i) for i in cropped_data]  # fft
fft_shifted = [np.fft.fftshift(i) for i in fft_data]  # fréquence ramené à > 0
fft_magnitude = [np.log(np.abs(i) + 1) for i in fft_shifted]  # log pour un meilleur contraste 
fft_magnitude_uint = [(i / i.max() * 255).astype(np.uint8) for i in fft_magnitude]  # conversion en uint8 pour cv2

# Threshold binaire pour isoler les pics
_, thresh_8 = cv2.threshold(fft_magnitude_uint[0], 200, 255, cv2.THRESH_BINARY)
_, thresh_9 = cv2.threshold(fft_magnitude_uint[1], 170, 255, cv2.THRESH_BINARY)
_, thresh_10 = cv2.threshold(fft_magnitude_uint[2], 175, 255, cv2.THRESH_BINARY)
thresh = [thresh_8, thresh_9, thresh_10]

peak_positions = []
central_peaks = []
distances_pixels = []
for i in range(len(thresh)):
    # Contours (pics) + coordonnées du centre des pics
    contour, _ = cv2.findContours(thresh[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = [cv2.moments(c) for c in contour]

    # Poistion des pics
    peak_position = [(int(c['m10']/c['m00']), int(c['m01']/c['m00'])) for c in center if c['m00'] != 0]
    peak_positions.append(peak_position)

    # Centre géométrique de l'image FFT
    center_x, center_y = fft_magnitude[i].shape[1] // 2, fft_magnitude[i].shape[0] // 2
    central_peaks.append((center_x, center_y))

    # Calcul des distances des pics par rapport au pic central
    distances_pixel = [np.sqrt((x - center_x)**2 + (y - center_x)**2) for x, y in peak_position]
    distances_pixels.append(distances_pixel)


# array
distances_pixels = [np.array(distances_pixels[i]) for i in range(len(distances_pixels))]


# Calcul des d_{hkl}
d_hkl = []
for i in range(len(distances_pixels)):
    d_hkl.append([(2*np.pi) / (d / (scale_pm_per_pixel*(len(cropped_data[i])/len(data[i])))) if d > 0 else None for d in distances_pixels[i]])

print('-- 8 --')
for i, (R, d_hkl_) in enumerate(zip(distances_pixels[0]/scale_pm_per_pixel, d_hkl[0])):
    print(f"Pic {i+1}: Distance en pm-1 = {R:.2f}, d_hkl = {d_hkl_} pm")
print('-- 9 --')
for i, (R, d_hkl_) in enumerate(zip(distances_pixels[1]/scale_pm_per_pixel, d_hkl[1])):
    print(f"Pic {i+1}: Distance en pm-1 = {R:.2f}, d_hkl = {d_hkl_} pm")
print('-- 10 --')
for i, (R, d_hkl_) in enumerate(zip(distances_pixels[2]/scale_pm_per_pixel, d_hkl[2])):
    print(f"Pic {i+1}: Distance en pm-1 = {R:.2f}, d_hkl = {d_hkl_} pm")


# Graphiques
fig, ax = plt.subplots(3, 3)
ax[0,0].imshow(data_8)
ax[0,1].imshow(data_9)
ax[0,2].imshow(data_10)

for i in range(len(cropped_data)):
    ax[1,i].imshow(cropped_data[i])

for i in range(len(fft_magnitude)):
    ax[2,i].imshow(fft_magnitude[i])

for i in range(len(peak_positions[0])):
    ax[2,0].scatter(peak_positions[0][i][0], peak_positions[0][i][1], color='r', s=3)
ax[2,0].scatter(central_peaks[0][0], central_peaks[0][1], color='black', s=3)
for i in range(len(peak_positions[1])):
    ax[2,1].scatter(peak_positions[1][i][0], peak_positions[1][i][1], color='r', s=3)
ax[2,1].scatter(central_peaks[1][0], central_peaks[1][1], color='black', s=3)
for i in range(len(peak_positions[2])):
    ax[2,2].scatter(peak_positions[2][i][0], peak_positions[2][i][1], color='r', s=3)
ax[2,2].scatter(central_peaks[2][0], central_peaks[2][1], color='black', s=3)

fig, ax = plt.subplots(1,3) # Juste les treillis zoomés
nb_pix_side = 50
for i in range(len(cropped_data)):
    ax[i].imshow(cropped_data[i][25:nb_pix_side+25, 25:nb_pix_side+25])
    bar_x = 25 # Position X de la barre
    bar_y = cropped_data[i][25:nb_pix_side+25, 25:nb_pix_side+25].shape[0]-5  # Position Y (bas de l'image)
    
    scale_bar = patches.Rectangle((bar_x, bar_y), nb_pix_scalebar, bar_height, color='white', linewidth=0)
    ax[i].add_patch(scale_bar)

    # Ajout du texte de la scale bar
    ax[i].text(bar_x + nb_pix_scalebar / 2, bar_y - 3, f'{scale_five_pixels:.2f} pm', 
               color='white', ha='center', fontsize=10)
plt.show()