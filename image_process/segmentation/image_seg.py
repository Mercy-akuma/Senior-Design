import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from helper import dir_file

 
from skimage.transform import (hough_line, hough_line_peaks, hough_circle,
hough_circle_peaks)
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.data import astronaut
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage import img_as_float
from skimage.morphology import skeletonize
from skimage import data, img_as_float
import matplotlib.pyplot as pylab
from matplotlib import cm
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, find_boundaries

from skimage import morphology
from scipy import ndimage as ndi


# coins = data.coins()
# hist = np.histogram(coins, bins=np.arange(0, 256), normed=True)
# fig, axes = pylab.subplots(1, 2, figsize=(20, 10))
# axes[0].imshow(coins, cmap=pylab.cm.gray, interpolation='nearest')
# axes[0].axis('off'), axes[1].plot(hist[1][:-1], hist[0], lw=2)
# axes[1].set_title('histogram of gray values')
# pylab.show()

path="../..//Figure/"
image_list=["test_rgb_image.jpg","test_thermal.png","test.jpg"]
for image in image_list:
    img_origin =cv2.imread(path+image)
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    ret1, coins= cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elevation_map = sobel(coins)
    fig, axes = pylab.subplots(figsize=(10, 6))
    axes.imshow(elevation_map, cmap=pylab.cm.gray, interpolation='nearest')
    axes.set_title('elevation map'), axes.axis('off'), pylab.show()
    
    markers = np.zeros_like(coins)
    markers[coins < 30] = 1
    markers[coins > 150] = 2
    print(np.max(markers), np.min(markers))
    fig, axes = pylab.subplots(figsize=(10, 6))
    a = axes.imshow(markers, cmap=plt.cm.hot, interpolation='nearest')
    plt.colorbar(a)
    axes.set_title('markers'), axes.axis('off'), pylab.show()
    
    segmentation = morphology.watershed(elevation_map, markers)
    fig, axes = pylab.subplots(figsize=(10, 6))
    axes.imshow(segmentation, cmap=pylab.cm.gray, interpolation='nearest')
    axes.set_title('segmentation'), axes.axis('off'), pylab.show()
    
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_coins, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_coins, image=coins)
    fig, axes = pylab.subplots(1, 2, figsize=(20, 6), sharey=True)
    axes[0].imshow(coins, cmap=pylab.cm.gray, interpolation='nearest')
    axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
    axes[1].imshow(image_label_overlay, interpolation='nearest')
    for a in axes:
        a.axis('off')
    pylab.tight_layout(), pylab.show()