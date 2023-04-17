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



# cv2.namedWindow("W0")
# cv2.imshow("W0", img_resize)
# cv2.waitKey(delay = 0)
# cv2.destroyAllWindows()


dir_file.mkdir("seg_fig")

dir_file.del_files("seg_fig")

path="../..//Figure/"
image_list=["test_rgb_image.jpg","test_thermal.png","test.jpg"]
for image in image_list:

    dir_file.mkdir("seg_fig/"+image + "/")
    dir_file.del_files("seg_fig/"+image + "/")
    
    img_origin = cv2.imread(path+image)
    
    # print( img_origin.)
    # img1 = cv2.resize(img_origin, dsize = None, fx = 0.5, fy = 0.5)
    img_resize = cv2.resize(img_origin, dsize = None, fx = 1, fy = 1)

    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    h, w = img_resize.shape[:2]
    print("resize: ",h, w)



    #图像进行二值化
    ##第一种阈值类型
    ret1, img1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    print(ret1)
    ##第二种阈值类型
    ret2, img2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    print(ret2)
    ##第三种阈值类型
    ret3, img3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    print(ret3)
    ##第四种阈值类型
    ret4, img4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    print(ret4)
    # ##第五种阈值类型
    # ret5, img5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    # print(ret5)
    #将所有阈值类型得到的图像绘制到同一张图中
    plt.rcParams['font.family'] = 'SimHei'       #将全局中文字体改为黑体
    figure = [img1, img2, img3, img4]
    title = ["threshold 1", "threshold 2", "threshold 3", "threshold 4"]
    for i in range(4):
        figure[i] = cv2.cvtColor(figure[i], cv2.COLOR_BGR2RGB)   #转化图像通道顺序，这一个步骤要记得
        plt.tight_layout()
        plt.subplot(2, 2, i+1)
        plt.imshow(figure[i])
        plt.title(title[i])   #添加标题
    plt.savefig("seg_fig/image_seg.jpg")  #保存图像，如果不想保存也可删去这一行
    # plt.show()


    #边缘检测之Sobel 算子
    img_Sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize = 5)


    #K-means均值聚类
    Z = img_resize.reshape((-1, 3))
    Z = np.float32(Z)      #转化数据类型
    c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 4
    ret, label, center = cv2.kmeans(Z, k, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_kmeans = res.reshape((img_resize.shape))

    #watershed, 分水岭算法
    
    ret1, img_black = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#（图像阈值分割，将背景设为黑色）
    # ret1, img_black = cv2.threshold(img_Sobel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#（图像阈值分割，将背景设为黑色）
    ##noise removal（去除噪声，使用图像形态学的开操作，先腐蚀后膨胀）
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_black, cv2.MORPH_OPEN, kernel, iterations = 2)
    # sure background area(确定背景图像，使用膨胀操作)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area（确定前景图像，也就是目标）
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    # Finding unknown region（找到未知的区域）
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret3, markers = cv2.connectedComponents(sure_fg)  #用0标记所有背景像素点
    # Add one to all labels so that sure background is not 0, but 1（将背景设为1）
    markers = markers+1
    ##Now, mark the region of unknown with zero（将未知区域设为0）
    markers[unknown == 255] = 0
    img_watershed=img_resize.copy()
    markers = cv2.watershed(img_watershed, markers)   #进行分水岭操作
    img_watershed[markers == -1] = [0, 0, 255]    #边界区域设为-1，颜色设置为红色

    #使用Canny边缘检测器
    img_origin =cv2.imread(path+image)
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    ret1, coins= cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    plt.figure()
    edges = canny(coins, sigma=2)
    # fig, axes = pylab.subplots(figsize=(10, 6))
    # axes.imshow(edges, cmap=pylab.cm.gray, interpolation='nearest')
    # axes.set_title('Canny detector'), axes.axis('off'), pylab.show()
    plt.imshow(edges)
    plt.savefig("seg_fig/"+image + "/edges.jpg") 

    plt.figure()
    fill_holes = ndi.binary_fill_holes(edges)
    # fig, axes = pylab.subplots(figsize=(10, 6))
    # axes.imshow(fill_coins, cmap=pylab.cm.gray, interpolation='nearest')
    # axes.set_title('filling the holes'), axes.axis('off'), pylab.show()
    plt.imshow(fill_holes)
    plt.savefig("seg_fig/"+image + "/fill_holes.jpg") 

    plt.figure()
    cleaned = morphology.remove_small_objects(fill_holes, 21)
    plt.imshow(cleaned )
    plt.savefig("seg_fig/"+image + "/cleaned .jpg")
    # fig, axes = pylab.subplots(figsize=(10, 6))
    # axes.imshow(coins_cleaned, cmap=pylab.cm.gray, interpolation='nearest')
    # axes.set_title('removing small objects'), axes.axis('off'), pylab.show()

    save_list=["img_origin","img_resize","img_gray","img_Sobel","img_kmeans","img_black","sure_bg","sure_fg","img_watershed"]

    for i in range(len(save_list)):
        saveFile = "seg_fig/"+image + "/" + save_list[i] + ".jpg"
        cv2.imwrite(saveFile, eval(save_list[i]))

