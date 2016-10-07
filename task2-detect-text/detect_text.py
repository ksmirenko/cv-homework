import cv2

base = cv2.imread("input.bmp")

gaussed = cv2.GaussianBlur(base, ksize=(7, 7), sigmaX=0)
laplaced = cv2.Laplacian(gaussed, ddepth=3, dst=0, ksize=9, scale=80)
result = cv2.convertScaleAbs(laplaced)

cv2.imwrite("output.bmp", result)
