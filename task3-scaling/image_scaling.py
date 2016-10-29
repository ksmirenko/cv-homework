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


# filters the image with an ideal low frequency filter
def filter_low_freq(x, m, freq):
    def gen_filter():
        fltr = np.zeros((m + 1, 3))
        for i in range(0, m + 1):
            d = i - m / 2
            for c in range(0, 3):
                fltr[i][c] = 0.5 * (1 - cos(2 * pi * i / m)) * sin(freq * d) / (pi * d) if d != 0 else np.nan
        return fltr

    f = gen_filter()
    n = len(x)
    res = np.zeros((n, 3))
    for n in range(0, n):
        for k in range(max(0, n - m), n):
            if np.isnan(f[n - k][0]):
                res[n] = x[k]
                break
            res[n] += x[k] * f[n - k]
    return res


# scale an image row (lossless)
def part2():
    filtered = filter_low_freq(row4, 4, pi / 4)
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

    filtered = [filter_low_freq(row, 4, pi / 4) for row in img]
    scaled_down = [scale(row, 1 / 32, 8) for row in filtered]
    scaled_up = [scale(row, 1 / 8, 32) for row in scaled_down]

    save_map(scaled_down, "fig5_1.png")
    save_map(scaled_up, "fig5_2.png")
    row_norms = [cv2.norm(scaled_up[i], img[i]) for i in range(0, 32)]
    open("norm_image_filt.txt", "w").write(str(cv2.norm(np.array(row_norms, dtype="u1"))))


part1()
part2()
part3()
