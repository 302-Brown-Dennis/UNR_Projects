# Dennis Brown
# Experiment 1
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mag_spectrum(noisy_image_np):

    f_transform = np.fft.fft2(noisy_image_np)
    f_transform_shifted = np.fft.fftshift(f_transform)
    img_magnitude_spectrum = np.log(1 + np.abs(f_transform_shifted))
    return f_transform_shifted, img_magnitude_spectrum

def band_reject(noisy_image_np):
    
    fft_image, magnitude_spectrum = mag_spectrum(noisy_image_np)
    rows, cols = noisy_image_np.shape
    center_row, center_col = rows // 2, cols // 2
    radius = 35
    width = 2

    # Create band-reject mask
    x = np.arange(rows).reshape(-1, 1)
    y = np.arange(cols).reshape(1, -1)
    distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    band_reject_mask = (distance <= (radius - width)) | (distance >= (radius + width))

    # Apply band-reject filter
    filtered_fft_band_reject = fft_image * band_reject_mask
    filter_magnitude_spectrum = np.log(1 + np.abs(filtered_fft_band_reject))
    # inverse FFT to get the filtered image
    ifft_shifted = np.fft.ifftshift(filtered_fft_band_reject)
    ifft_result = np.fft.ifft2(ifft_shifted)
    filtered_image_band_reject = np.abs(ifft_result)

    # used to just overlay the filter over the noisy image for visualization
    overlay_view = cv2.merge((noisy_image_np, noisy_image_np, noisy_image_np))
    overlay_view[band_reject_mask == 0] = [0, 0, 0]

    plt.figure(figsize=(12, 8))
    # noisy img
    plt.subplot(2, 3, 1)
    plt.title("Noisy Image", fontsize=10) 
    plt.imshow(noisy_image_np, cmap='gray')
    plt.axis("off")

    # noisy img with band filter
    plt.subplot(2, 3, 2)
    plt.title("Noisy Image with Band-Reject Filter", fontsize=10) 
    plt.imshow(overlay_view, cmap="gray")
    plt.axis("off")

    # noisy img spectrum
    plt.subplot(2, 3, 3)
    plt.title("Magnitude Spectrum of the Noisy Image", fontsize=10)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis("off")
    
    # filtered img
    plt.subplot(2, 3, 4)
    plt.title("Band-Reject Filtered Image", fontsize=10) 
    plt.imshow(filtered_image_band_reject, cmap="gray")
    plt.axis("off")

    # filtered img spectrum
    plt.subplot(2, 3, 5)
    plt.title("Magnitude Spectrum of the Filtered Image", fontsize=10) 
    plt.imshow(filter_magnitude_spectrum, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    output_path = f'CS474 Image Processing/Project4/Experiment1/OutputImages/band_reject_images.png'
    plt.savefig(output_path, dpi=300)

def notch_filter(noisy_image_np):

    fft_image, magnitude_spectrum = mag_spectrum(noisy_image_np)
    rows, cols = noisy_image_np.shape
    crow, ccol = rows // 2, cols // 2
    notch_mask = np.ones((rows, cols), dtype=np.float32)

    notch_radius = 2
    notches = [
        (crow - 16, ccol - 32),  # top
        (crow + 16, ccol + 32),  # bottom
        (crow + 16, ccol - 32),  # left
        (crow - 16, ccol + 32)   # right
    ]

    # Apply the notches
    for notch_center in notches:
        y, x = np.ogrid[:rows, :cols]
        mask_area = (y - notch_center[0])**2 + (x - notch_center[1])**2 <= notch_radius**2
        notch_mask[mask_area] = 0

    notch_filtered_spectrum = fft_image * notch_mask
    notch_filtered_image = np.fft.ifft2(np.fft.ifftshift(notch_filtered_spectrum)).real

    # used to just overlay the filter over the noisy image for visualization
    overlay_view = cv2.merge((noisy_image_np, noisy_image_np, noisy_image_np))
    overlay_view[notch_mask == 0] = [0, 0, 0]

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title("Noisy Image with Notch Filter", fontsize=10) 
    plt.imshow(overlay_view, cmap='gray')
    plt.axis("off")

    # Display the notch mask, filtered spectrum, and the filtered image
    plt.subplot(1, 3, 2)
    plt.title("Notch Filtered Image", fontsize=10) 
    plt.imshow(notch_filtered_image, cmap="gray")
    plt.axis("off")

    # Filtered spectrum
    plt.subplot(1, 3, 3)
    plt.title("Magnitude Spectrum of the Filtered Image", fontsize=10) 
    plt.imshow(np.log(1 + np.abs(notch_filtered_spectrum)), cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    output_path = f'CS474 Image Processing/Project4/Experiment1/OutputImages/notch_reject_images.png'
    plt.savefig(output_path, dpi=300)

def gauss_filter(noisy_image_np):

    gaussian_filtered_7x7 = cv2.GaussianBlur(noisy_image_np, (7, 7), 0)
    gaussian_filtered_15x15 = cv2.GaussianBlur(noisy_image_np, (15, 15), 0)

    # Display the Gaussian-filtered images
    plt.figure(figsize=(12, 8))

    # Original noisy image
    plt.subplot(1, 3, 1)
    plt.title("Noisy Image", fontsize=10) 
    plt.imshow(noisy_image_np, cmap="gray") 
    plt.axis("off")

    # Gaussian filtered (7x7)
    plt.subplot(1, 3, 2)
    plt.title("Gaussian Filtered 7x7", fontsize=10) 
    plt.imshow(gaussian_filtered_7x7, cmap="gray")
    plt.axis("off")

    # Gaussian filtered (15x15)
    plt.subplot(1, 3, 3)
    plt.title("Gaussian Filtered 15x15", fontsize=10) 
    plt.imshow(gaussian_filtered_15x15, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    output_path = f'CS474 Image Processing/Project4/Experiment1/OutputImages/gaussian_images.png'
    plt.savefig(output_path, dpi=300)

def noise_pattern(noisy_image_np):

    fft_image, magnitude_spectrum = mag_spectrum(noisy_image_np)
    rows, cols = noisy_image_np.shape
    crow, ccol = rows // 2, cols // 2
    notch_mask = np.ones((rows, cols), dtype=np.float32)

    notch_radius = 2
    notches = [
        (crow - 16, ccol - 32),  # top
        (crow + 16, ccol + 32),  # bottom
        (crow + 16, ccol - 32),  # left
        (crow - 16, ccol + 32)   # right
    ]

    # Apply the notches
    for notch_center in notches:
        y, x = np.ogrid[:rows, :cols]
        mask_area = (y - notch_center[0])**2 + (x - notch_center[1])**2 <= notch_radius**2
        notch_mask[mask_area] = 0

    # invert notch mask to extract pattern
    notch_mask_bool = notch_mask.astype(bool)
    inverted_mask = np.logical_not(notch_mask_bool)

    noise_pattern = fft_image * inverted_mask
    extracted_noise = np.abs(np.fft.ifft2(np.fft.fftshift(noise_pattern)))

    # apply FFT to extract spectrum of the noise
    extracted_noise_fft = np.fft.fft2(extracted_noise)
    extracted_noise_fft_shifted = np.fft.fftshift(extracted_noise_fft)
    extracted_noise_sepctrum = np.log(1 + np.abs(extracted_noise_fft_shifted))

    # Display the extracted noise pattern in both frequency and spatial domains
    plt.figure(figsize=(8, 8))

    # Extracted noise pattern
    plt.subplot(1, 2, 1)
    plt.title("Extracted Noise Pattern", fontsize=14) 
    plt.imshow(extracted_noise, cmap="gray")
    plt.axis("off")

    # Spectrum of the extracted noise
    plt.subplot(1, 2, 2)
    plt.title("Spectrum of Extracted Noise", fontsize=14) 
    plt.imshow(extracted_noise_sepctrum, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    output_path = f'CS474 Image Processing/Project4/Experiment1/OutputImages/extracted_noise_images.png'
    plt.savefig(output_path, dpi=300)

# Load the noisy image
image_path = 'CS474 Image Processing/Project4/Images/boy_noisy.pgm'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
noisy_image_np = np.array(img)

band_reject(noisy_image_np)
notch_filter(noisy_image_np)
gauss_filter(noisy_image_np)
noise_pattern(noisy_image_np)
plt.show()