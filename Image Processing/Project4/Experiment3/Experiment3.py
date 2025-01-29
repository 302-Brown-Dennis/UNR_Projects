# Dennis Brown
# Experiment 3
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = 'CS474 Image Processing/Project4/Images/girl.pgm'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
new_image_np = np.array(img)

# apply log to image
log_image = np.log1p(new_image_np)

# FFT and center the spectrum
fft_image = np.fft.fft2(log_image)
fft_image_shifted = np.fft.fftshift(fft_image)

# high-pass filter
rows, cols = new_image_np.shape
crow, ccol = rows // 2, cols // 2

# Filter parameters
D0 = 1.8       # cutoff frequency
gamma_L = 0.3  # low-frequency gain
gamma_H = 1.1  # high-frequency gain
c = 1          # constant

# generate the filter from the equation
u = np.arange(-ccol, ccol)
v = np.arange(-crow, crow)
D_squared = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        D_squared[i, j] = (u[j] if j < cols else u[j - cols])**2 + (v[i] if i < rows else v[i - rows])**2
high_pass_filter = (gamma_H - gamma_L) * (1 - np.exp(-c * D_squared / (D0**2))) + gamma_L

# apply the filter
filtered_fft = fft_image_shifted * high_pass_filter
filtered_fft_spectrum = filtered_fft
# inverse FFT on the image
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
filtered_image = np.expm1(filtered_image)

plt.figure(figsize=(12, 8))

# input image spectrum
plt.subplot(1, 2, 1)
plt.title("Input Image Spectrum", fontsize=16) 
plt.imshow(np.log(1 + np.abs(fft_image_shifted)), cmap="gray")
plt.axis("off")
# filtered image spectrum
plt.subplot(1, 2, 2)
plt.title("Filtered Image Spectrum", fontsize=16) 
plt.imshow(np.log(1 + np.abs(filtered_fft_spectrum)), cmap="gray")
plt.axis("off")
plt.tight_layout()

output_path = f'CS474 Image Processing/Project4/Experiment3/OutputImages/image_spectrums.png'
plt.savefig(output_path, dpi=300)

plt.figure(figsize=(8, 8))
# input image
plt.subplot(1, 3, 1)
plt.title("Input Image", fontsize=16) 
plt.imshow(new_image_np, cmap="gray")
plt.axis("off")

# homomorphic filter
plt.subplot(1, 3, 2)
plt.title("High-Pass Filter", fontsize=16) 
plt.imshow(high_pass_filter, cmap="gray")
plt.axis("off")

# filtered image
plt.subplot(1, 3, 3)
plt.title("Filtered Image", fontsize=16) 
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")
plt.tight_layout()

output_path = f'CS474 Image Processing/Project4/Experiment3/OutputImages/filtered_images.png'
plt.savefig(output_path, dpi=300) 

plt.show()