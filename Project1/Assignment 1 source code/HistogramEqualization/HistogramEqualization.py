# soruces citied:
# https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html
# Dennis Brown
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

image_path = ['CS474/HW1/Images/f_16.pgm', 'CS474/HW1/Images/boat.pgm']
image_titles = ['f_16', 'Boat']

for current_img, title in zip(image_path, image_titles):
    img = cv2.imread(current_img, cv2.IMREAD_GRAYSCALE)

    # calculate intensity distributions
    hist = np.zeros(256)
    for pixel in img.flatten():
        hist[pixel] += 1

    # calculate cdf and cumulative sum
    cdf = hist.cumsum()
    # normalize our cdf and apply to input image
    cdf_normalized = cdf * (255 / cdf[-1])
    equalized_image = np.interp(img, range(0, 256), cdf_normalized)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f'{title} Original Image')

    plt.figure(figsize=(8, 8))
    plt.title(f'{title} Original Histogram')
    plt.xlim([0, 256])
    output_path = f'CS474/HW1/HistogramEqualization/OutputImages/{title}_Original_Histogram_Output_image.png'
    plt.savefig(output_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(equalized_image, cmap='gray')
    plt.title(f'{title} Equalized Image')
    output_path = f'CS474/HW1/HistogramEqualization/OutputImages/{title}_Equalized_Image_Output_image.png'
    cv2.imwrite(output_path, equalized_image) 

    plt.figure(figsize=(8, 8))
    plt.hist(equalized_image.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.title(f'{title} Equalized Histogram')
    output_path = f'CS474/HW1/HistogramEqualization/OutputImages/{title}_Equalizied_Histogram_Output_image.png'
    plt.savefig(output_path)

plt.show()

