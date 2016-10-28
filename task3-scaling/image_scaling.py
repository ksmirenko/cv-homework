from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plot
from math import sin, cos, pi

img = cv2.imread("chessboard.bmp")
row4 = np.array(img, dtype='u1')[4]


# performs sampling of a continuous signal
def sampling(y, t, n):  # signal, period, number of components
    result = np.zeros((n, 3), dtype='u1')
    for i in range(0, n):
        result[i] = y(i * t)
    return result


# restores continuous signal from original image row
def cont_signal(x, t_s, m):  # input, period, degree of approximation
    def signal(t):
        s = np.zeros(3, dtype='u1')
        lower = int((-m / 2 + t) / t_s) + 1
        upper = int((m / 2 + t) / t_s)
        for n in range(lower, upper):
            index = n % len(x)
            divider = pi * (t - n * t_s)
            for i in range(0, 3):
                s[i] += sin(divider / t_s) * x[index][i] / divider if divider != 0 else x[index][i]
        return s

    return signal


# performs scaling of a 1D image row
def scale(x, t_old, k):  # input, period, output number of components
    y = cont_signal(x, t_old, 4)
    return sampling(y, 1 / k, k)


def save_plot(source, filename):
    plot.plot(source, linewidth=2.0)
    plot.savefig(filename)
    plot.close()


def save_map(source, filename):
    plot.imshow(source)
    plot.savefig(filename)
    plot.close()


# scale an image row (lossy)
def part1():
    scaled_down = scale(row4, 1 / 32, 8)
    scaled_up = scale(scaled_down, 1 / 8, 32)
    save_plot(scaled_down, "fig2_1.png")
    save_plot(scaled_up, "fig2_2.png")
    open("norm_row.txt", "w").write(str(cv2.norm(row4, scaled_up)))


# generates an ideal low frequency filter with given frequency
def low_freq_filter(freq, m):
    result = np.zeros((m, 3))
    for n in range(0, m):
        k = n - m / 2
        w = 0.5 - 0.5 * cos(2 * pi * n / m) if 0 <= n <= m else 0
        for i in range(0, 3):
            result[n][i] = sin(freq * k) / (pi * k) * w if k != 0 else 0
    return result


# scale an image row (lossless)
def part2():
    fft1 = abs(np.fft.fft(row4))
    fft2 = abs(np.fft.fft(low_freq_filter(pi / 4, len(row4))))
    filtered = abs(np.fft.ifft(fft1 * fft2)).astype(int)

    scaled_down = scale(filtered, 1 / 32, 8)
    scaled_up = scale(scaled_down, 1 / 8, 32)
    save_plot(scaled_down, "fig3_1.png")
    save_plot(scaled_up, "fig3_2.png")
    open("norm_row_filt.txt", "w").write(str(cv2.norm(row4, scaled_up)))


def part3():
    scaled_down = [scale(row, 1 / 32, 8) for row in img]
    scaled_up = [scale(row, 1 / 8, 32) for row in scaled_down]

    save_map(scaled_down, "fig4_1.png")
    save_map(scaled_up, "fig4_2.png")
    row_norms = [cv2.norm(scaled_up[i], img[i]) for i in range(0, 32)]
    open("norm_image.txt", "w").write(str(cv2.norm(np.array(row_norms, dtype="u1"))))

    fltr = low_freq_filter(pi / 4, len(img[0]))
    fft2 = abs(np.fft.fft(fltr))
    filtered = [None] * 32
    for i in range(0, 32):
        fft1 = abs(np.fft.fft(img[i]))
        filtered[i] = abs(np.fft.ifft(fft1 * fft2)).astype(int)

    scaled_down = [scale(row, 1 / 32, 8) for row in filtered]
    scaled_up = [scale(row, 1 / 8, 32) for row in scaled_down]

    save_map(scaled_down, "fig5_1.png")
    save_map(scaled_up, "fig5_2.png")
    row_norms = [cv2.norm(scaled_up[i], img[i]) for i in range(0, 32)]
    open("norm_image_filt.txt", "w").write(str(cv2.norm(np.array(row_norms, dtype="u1"))))


part1()
part2()
part3()
