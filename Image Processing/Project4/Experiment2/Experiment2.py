# Dennis Brown
# Experiment 2
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = 'CS474 Image Processing/Project4/Images/lenna.pgm'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_np = np.array(img)

# Sobel kernel
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Pad image with 0's
f_p = np.pad(image_np, ((1, 1), (1, 1)), mode='constant', constant_values=0)
f_p = f_p.astype(np.int32)

# center spectrum
for x in range(f_p.shape[0]):
    for y in range(f_p.shape[1]):
        f_p[x, y] *= (-1) ** (x + y)
# compute FFT
f_p_fft = np.fft.fft2(f_p)

# pad kernel with leading 0 row and col
h_p_kernel_pad = np.pad(sobel_kernel, ((1, 0), (1, 0)), mode="constant", constant_values=0)

# make new kernel with size of padded image
h_p = np.zeros((f_p.shape[0], f_p.shape[1]))

# find center
center_row = f_p.shape[0]//2
center_col = f_p.shape[1]//2

start_row = center_row - 2
start_col = center_col - 2

# insert padded kernel
h_p[start_row:start_row + 4, start_col:start_col + 4] = h_p_kernel_pad

# center spectrum
for u in range(h_p.shape[0]):
    for v in range(h_p.shape[1]):
        h_p[u, v] *= (-1) ** (u + v)

# compute FFT, set real part to 0
h_p_fft = np.fft.fft2(h_p)
h_p_fft.imag = h_p_fft.imag
h_p_fft.real = 0

# undo centering
for u in range(h_p_fft.shape[0]):
    for v in range(h_p_fft.shape[1]):
        h_p_fft[u, v] *= (-1) ** (u + v)

sobel_imag_part = h_p_fft.imag

# only keep real part
G_u = (h_p_fft * f_p_fft)
g_u = np.fft.ifft2(G_u).real

# undo centering one last time
for x in range(g_u.shape[0]):
    for y in range(g_u.shape[1]):
        g_u[x, y] *= (-1) ** (x + y)

# trim padding
g_u = g_u[1:-1, 1:-1]
gu_scaled = (g_u - np.min(g_u)) / (np.max(g_u) - np.min(g_u)) * 255

# flip sobel kernel, apply and scale values
kernel_flip = np.flipud(np.fliplr(sobel_kernel))
sobel_spatial = cv2.filter2D(image_np, ddepth=cv2.CV_64F, kernel=kernel_flip)
sobel_spatial_scaled = (sobel_spatial - np.min(sobel_spatial)) / (np.max(sobel_spatial) - np.min(sobel_spatial)) * 255
sobel_spec = cv2.filter2D(image_np, -1, kernel=kernel_flip)
sobel_spatial_spectrum = np.fft.fftshift(np.fft.fft2(sobel_spec))

plt.figure(figsize=(12, 8))
# Display the input image
plt.subplot(2, 3, 1)
plt.title("Input Image")
plt.imshow(image_np, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Frequency Sobel Filter", fontsize=12) 
plt.imshow(gu_scaled, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Spatial Sobel Filter", fontsize=12) 
plt.imshow(sobel_spatial_scaled, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Input Image Spectrum", fontsize=12) 
plt.imshow(np.log(1 + np.abs(f_p_fft)), cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Frequency Sobel Spectrum", fontsize=12) 
plt.imshow(np.log(1 + np.abs(G_u)), cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Spatial Sobel Spectrum", fontsize=12) 
plt.imshow(np.log(1 + np.abs(sobel_spatial_spectrum)), cmap="gray")
plt.axis("off")
plt.tight_layout()

output_path = f'CS474 Image Processing/Project4/Experiment2/OutputImages/sobel_filter_images.png'
plt.savefig(output_path, dpi=300)

# show transfer func
plt.figure(figsize=(6, 6))
plt.title("Sobel Transfer Function Spectrum", fontsize=14) 
plt.imshow(sobel_imag_part, cmap="gray")
plt.axis("off")
plt.tight_layout()

output_path = f'CS474 Image Processing/Project4/Experiment2/OutputImages/transfer_function_image.png'
plt.savefig(output_path, dpi=300)

plt.show()