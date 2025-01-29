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

rect_data = np.loadtxt('/VS Code WorkSpace/CS474 Image Processing/Project3/Rect_128.txt')

plt.figure(figsize=(10, 4))
plt.plot(rect_data)
plt.title("Rectangular Function")
plt.grid()
plt.savefig("CS474 Image Processing/Project3/Experiment1/1c/1cOutputRect.png", dpi=300, bbox_inches="tight")
N = len(rect_data)
data = np.zeros(2 * N + 1)
data[1::2] = rect_data

fft(data, N, -1)
data_normalized = (1/N) * data

real_part = data_normalized[1::2]
imaginary_part = data_normalized[:-1:2]
magnitude = np.sqrt(real_part**2 + imaginary_part**2)
phase = np.arctan2(imaginary_part, real_part)

half_N = N // 2
shifted_magnitude = np.concatenate((magnitude[half_N:], magnitude[:half_N]))

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.stem(real_part)
plt.title("Real Part of DFT")

plt.subplot(4, 1, 2)
plt.stem(imaginary_part)
plt.title("Imaginary Part of DFT")

plt.subplot(4, 1, 3)
plt.stem(shifted_magnitude)
plt.title("Shifted Magnitude of DFT")

plt.subplot(4, 1, 4)
plt.stem(phase)
plt.title("Phase of DFT")

plt.tight_layout()
plt.savefig("CS474 Image Processing/Project3/Experiment1/1c/1cOutput.png", dpi=300, bbox_inches="tight")
plt.show()