from __future__ import division
import cv2
import os
from itertools import chain

cascade = cv2.CascadeClassifier("data/cascade/cascade.xml")


def test_suite(paths):
    all_tests = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
        for _file in files:
            is_square = _file.startswith("square-")
            image = cv2.imread(root + "/" + _file)
            answer = len(cascade.detectMultiScale(image, scaleFactor=1.3)) > 0
            if is_square and answer:
                true_positives += 1
            if not is_square and not answer:
                true_negatives += 1
            if not is_square and answer:
                false_positives += 1
            if is_square and not answer:
                false_negatives += 1
            all_tests += 1

    print("TEST SUITE")
    print("Tests count: {}".format(all_tests))
    print("Accuracy:\t{0:.3f}%".format((true_positives + true_negatives) * 100 / all_tests))
    print("Precision:\t{0:.3f}%".format(true_positives * 100 / (true_positives + false_positives)))
    print("Recall:\t\t{0:.3f}%".format(true_positives * 100 / (true_positives + false_negatives)))
    print("True positives:  {}".format(true_positives))
    print("True negatives:  {}".format(true_negatives))
    print("False positives: {}".format(false_positives))
    print("False negatives: {}".format(false_negatives))
    print("")


test_suite(["data/squares", "data/triangles"])
test_suite(["data/tests"])
