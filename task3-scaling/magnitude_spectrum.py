import cv2
import numpy as np
import matplotlib.pyplot as plot

img = cv2.imread("chessboard.bmp")
row = img[4]

spectrum = abs(np.fft.fft(row))
plot.plot(spectrum, linewidth=2.0)
plot.show()

spectrum2 = abs(np.fft.fft2(img))
plot.imshow(spectrum2)
plot.show()
