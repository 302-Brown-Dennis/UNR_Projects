import numpy as np
import matplotlib.pyplot as plt

# FFT function from the provided code
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

f = np.array([2, 3, 4, 4])
nn = len(f)

data = np.zeros(2 * nn + 1)
# Set real part
data[1::2] = f

fft(data, nn, -1)

data_normalized = (1/nn) * data

real_part = data_normalized[1::2]
print("Real part: ", real_part)

imaginary_part = data_normalized[:-1:2]
print("imaginary part: ", imaginary_part)

magnitude = np.sqrt(real_part**2 + imaginary_part**2)
print("magnitude ", magnitude)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(real_part)
plt.title("Real Part of DFT")
plt.subplot(3, 1, 2)
plt.stem(imaginary_part)
plt.title("Imaginary Part of DFT")
plt.subplot(3, 1, 3)
plt.stem(magnitude)
plt.title("Magnitude of DFT")
plt.tight_layout()

original_signal = data_normalized[1::2]
print("Inverse FFT: ", original_signal)
plt.savefig("CS474 Image Processing/Project3/Experiment1/1a/1aOutput.png", dpi=300, bbox_inches="tight")
plt.show()