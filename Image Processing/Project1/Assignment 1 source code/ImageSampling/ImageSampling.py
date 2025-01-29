# Dennis Brown
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

image_path = ['CS474/HW1/Images/lenna.pgm', 'CS474/HW1/Images/peppers.pgm']
image_titles = ['lenna', 'peppers']

for current_img, title in zip(image_path, image_titles):
  img = cv2.imread(current_img, cv2.IMREAD_GRAYSCALE)
   #Display the original image
  plt.figure(figsize=(8, 8))
  plt.imshow(img, cmap='gray')
  plt.title(f'{title} Original Image')
  # Sample image by factor of 2
  two_img_sample = np.zeros((128,128,3), dtype=np.uint8)
  newj = 0
  newi = 0
  # map pixels to new image
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if j % 2 != 0:
          newj = (math.trunc(j/2))
          newi = (math.trunc(i/2))
          two_img_sample[newi][newj] = img[i][j]
  # upscale image to 256x256
  two_img_sample_final = np.zeros((256,256,3), dtype=np.uint8)
  for i in range(0,128):
    for j in range(0,128):
      pixval = two_img_sample[i, j]
      two_img_sample_final[2 * i, 2 * j] = pixval
      two_img_sample_final[2 * i + 1, 2 * j] = pixval
      two_img_sample_final[2 * i, 2 * j + 1] = pixval
      two_img_sample_final[2 * i + 1, 2 * j + 1] = pixval

  # Sample image by factor of 4
  four_img_sample = np.zeros((64,64,3), dtype=np.uint8)
  newj = 0
  newi = 0
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if j % 4 != 0:
          newj = (math.trunc(j/4))
          newi = (math.trunc(i/4))
          four_img_sample[newi][newj] = img[i][j]

  four_img_sample_final = np.zeros((256,256,3), dtype=np.uint8)
  for i in range(0,64):
    for j in range(0,64):
      pixval = four_img_sample[i,j]
      # 4x4 window
      for k in range(4):
          for l in range(4):
              four_img_sample_final[4 * i + k, 4 * j + l] = pixval

  # Sample image by factor of 8
  eight_img_sample = np.zeros((32,32,3), dtype=np.uint8)

  newj = 0
  newi = 0
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if j % 8 != 0:
          newj = (math.trunc(j/8))
          newi = (math.trunc(i/8))
          eight_img_sample[newi][newj] = img[i][j]

  eight_img_sample_final = np.zeros((256,256,3), dtype=np.uint8)
  for i in range(0,32):
    for j in range(0,32):
      pixval = eight_img_sample[i,j]
      # 8x8 window
      for k in range(8):
          for l in range(8):
              eight_img_sample_final[8 * i + k, 8 * j + l] = pixval

  plt.figure(figsize=(8, 8))
  plt.imshow(two_img_sample_final, cmap='gray')
  plt.title(f'{title} 2x Smaple')


  plt.figure(figsize=(8, 8))
  plt.imshow(four_img_sample_final, cmap='gray')
  plt.title(f'{title} 4x sample')

  plt.figure(figsize=(8, 8))
  plt.imshow(eight_img_sample_final, cmap='gray')
  plt.title(f'{title} 8x sample')

  output_path = f'CS474/HW1/ImageSampling/OutputImages/{title}_2_SubSample_image.png'
  cv2.imwrite(output_path, two_img_sample_final) 
  output_path = f'CS474/HW1/ImageSampling/OutputImages/{title}_4_SubSample_image.png'
  cv2.imwrite(output_path, four_img_sample_final) 
  output_path = f'CS474/HW1/ImageSampling/OutputImages/{title}_8_SubSample_image.png'
  cv2.imwrite(output_path, eight_img_sample_final) 

plt.show()