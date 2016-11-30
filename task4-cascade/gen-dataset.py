from __future__ import division

import os
import cv2
import numpy as np
from math import sqrt, pi, sin, cos
from random import randint

squares_folder = "squares/"
triangles_folder = "triangles/"
squares_path = "./data/" + squares_folder
triangles_path = "./data/" + triangles_folder
tests_path = "./data/tests/"
cascade_path = "./data/cascade/"

# Make dirs if missing
for path in [squares_path, triangles_path, tests_path, cascade_path]:
    if not os.path.exists(path):
        os.makedirs(path)

img_size = 50
train_data_size = 800
test_data_size = 200


# Generates a square
def generate_square(filename):
    image = np.ones((img_size, img_size, 3)) * 255

    edge_length = randint(int(img_size / 5), int(img_size / 2 * sqrt(2)))
    gap = int(edge_length / sqrt(2))
    x0 = randint(gap, img_size - gap)
    y0 = randint(gap, img_size - gap)
    alpha = randint(0, 90) * pi / 180.0
    h = int(edge_length / 2)

    def turn(x, y):
        return [x0 + int(x * cos(alpha) - y * sin(alpha)),
                y0 + int(x * sin(alpha) + y * cos(alpha))]

    square = np.array([turn(-h, -h), turn(h, -h), turn(h, h), turn(-h, h)])
    color = (randint(0, 255), randint(0, 255), randint(0, 255))

    cv2.fillPoly(image, [square], color)
    cv2.imwrite(filename, image)


# Generates a triangle
def generate_triangle(filename):
    image = np.ones((img_size, img_size, 3)) * 255

    lower = int(img_size * 0.2)
    upper = int(img_size * 0.7)
    a = [randint(0, img_size), randint(0, lower)]
    b = [randint(0, lower), randint(upper, img_size)]
    c = [randint(upper, img_size), randint(upper, img_size)]

    triangle = np.array([a, b, c])
    color = (randint(0, 255), randint(0, 255), randint(0, 255))

    cv2.fillPoly(image, [triangle], color)
    cv2.imwrite(filename, image)


# Training data
for i in range(0, train_data_size):
    generate_square(squares_path + "square-" + str(i) + ".bmp")
    generate_triangle(triangles_path + "triangle-" + str(i) + ".bmp")

# Test data
for i in range(0, test_data_size):
    generate_square(tests_path + "square-" + str(i) + ".bmp")
    generate_triangle(tests_path + "triangle-" + str(i) + ".bmp")

# .dat files
# we consider squares "positive" and triangles "negative"
square_dat = open("./data/squares.dat", "w+")
for i in range(0, train_data_size):
    square_dat.write("{0}square-{1}.bmp 1 0 0 {2} {2}\n".format(squares_folder, i, img_size))
square_dat.close()

triangle_dat = open("./data/triangles.dat", "w+")
for i in range(0, train_data_size):
    triangle_dat.write("{0}triangle-{1}.bmp\n".format(triangles_folder, i))
triangle_dat.close()
