import numpy as np
import skimage
from skimage.draw import polygon
from skimage.draw import bezier_curve
from skimage.draw import circle_perimeter
from skimage.draw import disk
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_color_gradient(percent, color='Blues'):
    return mpl.colormaps[color](percent)[:3]


def agt_color(aidx):
    return matplotlib.colors.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][aidx % 10])


def draw_disk(img_size=80, max_r=10, iterations=3):
    shape = (img_size, img_size)
    img = np.zeros(shape, dtype=np.uint8)
    x, y = np.random.uniform(max_r, img_size - max_r, size=(2))
    radius = int(np.random.uniform(max_r))
    rr, cc = disk((x, y), radius, shape=shape)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_circle(img_size=80, max_r=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r, c = np.random.uniform(max_r, img_size - max_r, size=(2,)).astype(int)
    radius = int(np.random.uniform(max_r))
    rr, cc = circle_perimeter(r, c, radius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=1).astype(int)
    return img


def draw_curve(img_size=80, max_sides=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r0, c0, r1, c1, r2, c2 = np.random.uniform(0, img_size, size=(6,)).astype(int)
    w = np.random.random()
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, w)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=iterations).astype(int)
    return img


def draw_polygon(img_size=80, max_sides=10):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    num_coord = int(np.random.uniform(3, max_sides))
    r = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    c = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    rr, cc = polygon(r, c)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_ellipse(img_size=80, max_size=10):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r, c, rradius, cradius = np.random.uniform(max_size, img_size - max_size), np.random.uniform(max_size, img_size - max_size),\
        np.random.uniform(1, max_size), np.random.uniform(1, max_size)
    rr, cc = skimage.draw.ellipse(r, c, rradius, cradius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img