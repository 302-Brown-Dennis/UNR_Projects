# Dennis Brown
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

image_path = ['CS474/HW1/Images/lenna.pgm', 'CS474/HW1/Images/peppers.pgm']
image_titles = ['lenna', 'peppers']

# store gray level and max pixel value
gray_levels = [128, 32, 8, 2]
max_pixel_intensity = 255
for current_img, title in zip(image_path, image_titles):
    img = cv2.imread(current_img, cv2.IMREAD_GRAYSCALE)
    print(img)
    # Display the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f'{title} Original Image')
    for level in gray_levels:
        # calculate L
        stepper = max_pixel_intensity // level
        # apply to pixel values and sub sample gray levels
        quantize_img = (img // stepper) * stepper
        plt.figure(figsize=(8, 8))
        plt.imshow(quantize_img, cmap='gray')
        plot_title = f'L={math.trunc(level)} Quantized Image'
        plt.title(plot_title)
        output_path = f'CS474/HW1/ImageQuantization/OutputImages/{title}_L_{level}_Quantized_Output_image.png'
        cv2.imwrite(output_path, quantize_img) 

plt.show()