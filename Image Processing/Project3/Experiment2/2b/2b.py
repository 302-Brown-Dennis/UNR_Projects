import numpy as np
import matplotlib.pyplot as plt

def fft(data, nn, isign):
    n = nn * 2
    j = 0
    
    for i in range(0, n, 2):
        if j > i:
            data[j], data[i] = data[i], data[j]
            data[j + 1], data[i + 1] = data[i + 1], data[j + 1]
        m = n // 2
        while m >= 2 and j >= m:
            j -= m
            m //= 2
        j += m
    
    mmax = 2
    while n > mmax:
        istep = mmax * 2
        theta = isign * (2 * np.pi / mmax)
        wtemp = np.sin(0.5 * theta)
        wpr = -2.0 * wtemp * wtemp
        wpi = np.sin(theta)
        wr = 1.0
        wi = 0.0

        for m in range(0, mmax, 2):
            for i in range(m, n, istep):
                j = i + mmax
                tempr = wr * data[j] - wi * data[j + 1]
                tempi = wr * data[j + 1] + wi * data[j]
                data[j] = data[i] - tempr
                data[j + 1] = data[i + 1] - tempi
                data[i] += tempr
                data[i + 1] += tempi
            wr, wi = (wr * wpr - wi * wpi + wr), (wi * wpr + wr * wpi + wi)

        mmax = istep

def fft2D(image, N, M, isign):
    
    real_part = np.copy(image).astype(float)
    imag_part = np.zeros_like(real_part)

    # 1D FFT on rows
    for i in range(N):
        data = np.zeros(2 * M + 1)
        data[1::2] = real_part[i, :]
        data[:-1:2] = imag_part[i, :]
        fft(data, M, isign)
        data_normalized = (1/M) * data
        real_part[i, :] = data_normalized[1::2]
        imag_part[i, :] = data_normalized[:-1:2]

    # 1D FFT on columns
    for j in range(M):
        data = np.zeros(2 * N + 1)
        data[1::2] = real_part[:, j]
        data[:-1:2] = imag_part[:, j]
        fft(data, N, isign)
        data_normalized = (1/N) * data
        real_part[:, j] = data_normalized[1::2]
        imag_part[:, j] = data_normalized[:-1:2]

    return real_part, imag_part

image_size = 512
square_size = 64
image = np.zeros((image_size, image_size), dtype=np.uint8)

start = (image_size - square_size) // 2
end = start + square_size

image[start:end, start:end] = 255

N, M = image.shape
real_dft, imag_dft = fft2D(image, N, M, isign=-1)

magnitude = np.sqrt(real_dft**2 + imag_dft**2)

checkerboard = np.ones((image_size, image_size), dtype=np.float32)
for i in range(image_size):
    for j in range(image_size):
        if (i + j) % 2 != 0:
            checkerboard[i, j] = -1
            
shifted_image = image * checkerboard
real_shifted_dft, imag_shifted_dft = fft2D(shifted_image, N, M, isign=-1)
shifted_magnitude = np.sqrt(real_shifted_dft**2 + imag_shifted_dft**2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Generated image with white Square")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("unshifted magnitude of DFT ")
plt.imshow(np.log(1 + magnitude), cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(np.log(1 + shifted_magnitude), cmap='gray')
plt.title("Shifted centered magnitude of DFT")
plt.axis('off')
plt.savefig("CS474 Image Processing/Project3/Experiment2/2b/2bOutput.png", dpi=300, bbox_inches="tight")
plt.show()