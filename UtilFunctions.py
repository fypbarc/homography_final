# Import necessary libraries
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pyautogui as p
import math
import sys
import pandas as pd


def plot_image(img):
    """Plots the image provided in a new window"""
    plt.imshow(img, cmap="viridis")
    plt.show()


def plot_images(images, figsize_in_inches=(5, 5)):
    """Plots all images provided in a subplot"""
    fig, axs = plt.subplots(1, len(images), figsize=figsize_in_inches)
    for col, image in enumerate(images):
        axs[col].imshow(image)
    plt.show()


def save_image(image: np.ndarray, save_file_name=None):
    """Save image using opencv"""
    if save_file_name is None:
        save_file_name = p.prompt(text="Save image name without an extension", title="Enter input image name")
    cv.imwrite(f"{save_file_name}.png", image)
    sys.stdout.write(f"Image successfully saved as {save_file_name}.png\n\n")


def get_expected_path_from_conf(conf_dict, raw_range: bool = False):
    """Creates a list of paths of images and returns in ordered format
    ie, [r1_c1, r1_c2, r1_c3.......r2_c1] r:rows & c:cols
    arguments:
    raw_range: if set to true instead of returning .png strings it will return a tuple of (x, y) cordinates, defaultsto fase"""
    x_start = float(conf_dict["X_RANGE"][0])
    x_stop = float(conf_dict["X_RANGE"][1])
    x_step = float(conf_dict["X_RANGE"][2])

    y_start = float(conf_dict["Y_RANGE"][0])
    y_stop = float(conf_dict["Y_RANGE"][1])
    y_step = float(conf_dict["Y_RANGE"][2])

    x_range_expec, y_range_expec = [], []

    if x_step == 0 or x_start == x_stop:
        x_range_expec = [x_start]
    else:
        x_range_expec = np.arange(x_start, x_stop, x_step)

    if y_step == 0 or y_start == y_stop:
        y_range_expec = [y_start]
    else:
        y_range_expec = np.arange(y_start, y_stop, y_step)

    if raw_range:
        return x_range_expec, y_range_expec

    paths_expec = []
    for _x in x_range_expec:
        x = f"{_x:.2f}"
        for _y in y_range_expec:
            y = f"{_y:.2f}"
            _path = x + "_" + y + ".png"
            paths_expec.append(_path)

    return paths_expec


def resize_images(images, size_factor=0.5):
    """This function resizes all input images within themselves.
    arguments:
    images: Either a single ndarray or list of ndarray.
    size_factor: Must be a float between 0 and 1. By what perfect an image has to be resized. Defaults to 50% ie. 0.5
    No returns."""

    if isinstance(images, list):
        for idx in range(len(images)):
            images[idx] = cv.resize(images[idx], None, fx=size_factor, fy=size_factor, interpolation=cv.INTER_LINEAR_EXACT)
    else:
        return cv.resize(images, None, fx=size_factor, fy=size_factor, interpolation=cv.INTER_LINEAR_EXACT)


def find_thresh_conf_matrix(conf_matrix):
    """Since the matches confidence is a list of confidences of each image with all other images we find a threshold
    that full remove all other matches except the immediate neighbour and returns it
    arguments:
    conf_matrix: a list of confidences
    returns:
    conf_threshold: calculated threshold
    """
    CONF_THRESH = float('inf')

    for idx in range(len(conf_matrix)):
        prev_id = idx - 1
        if prev_id < 0:
            prev_id = None
        thresh = conf_matrix[idx][prev_id] if prev_id is not None else float('inf')
        if CONF_THRESH > thresh:
            CONF_THRESH = math.floor(thresh * 1000) / 1000
    return CONF_THRESH


def get_confidence_matrix(matches):
    matrix_dimension = int(math.sqrt(len(matches)))
    rows = []
    for i in range(0, len(matches), matrix_dimension):
        rows.append(matches[i: i + matrix_dimension])
    match_confs = [[m.confidence for m in row] for row in rows]
    match_conf_matrix = np.array(match_confs)
    return match_conf_matrix


