import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from helper import dir_file



def addWeightedSmallImgToLargeImg(largeImg,alpha,smallImg,beta,gamma=0.0,regionTopLeftPos=(0,0)):
    srcW, srcH = largeImg.shape[1::-1]
    refW, refH = smallImg.shape[1::-1]
    x,y =  regionTopLeftPos
    if (refW>srcW) or (refH>srcH):
        #raise ValueError("img2's size must less than or equal to img1")
        raise ValueError(f"img2's size {smallImg.shape[1::-1]} must less than or equal to img1's size {largeImg.shape[1::-1]}")
    else:
        if (x+refW)>srcW:
            x = srcW-refW
        if (y+refH)>srcH:
            y = srcH-refH
        destImg = np.array(largeImg)
        tmpSrcImg = destImg[y:y+refH,x:x+refW]
        tmpImg = cv2.addWeighted(tmpSrcImg, alpha, smallImg, beta,gamma)
        destImg[y:y + refH, x:x + refW] = tmpImg
        return destImg



def appendimages(im1,im2,size=0):
    """ 返回将两幅图像并排拼接成的一幅新图像 """
    # 选取具有最少行数的图像，然后填充足够的空行
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        if size==0:
            im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
        else:
            im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1],size))),axis=0)
    elif rows1 > rows2:
        if size ==0:
            im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
        else:
            im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1],size))),axis=0)
    return np.concatenate((im1,im2), axis=1)

def gen_start_pos():
    img1_x=0
    img1_y=0
    img2_x=0
    img2_y=0
    return img1_x-img2_x,img1_y-img2_y

# def image_combine
    
    
img = cv2.imread("../../Figure/test_rgb_image.jpg")
dst1 = np.float32([(0,0), (20,20), (40,0)])
dst2 = np.float32([(20,10), (30,20), (50,20)])
# dst1=dst2
M = cv2.getAffineTransform(dst1, dst2)
dst_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cat_img = appendimages(img, dst_img, 3)


img_rgb = cv2.imread("../../Figure/test_rgb_image.jpg")
img_thermal = cv2.imread("../../Figure/test_thermal.png")
# img_thermal0 = cv2.imread("../../Figure/test.jpg")

img_shape=(img_rgb.shape[1], img_rgb.shape[0])
print(img_shape)


size=(img_thermal.shape[1], img_thermal.shape[0])
img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)

a=0.5 # Weighted
b=1 # Weighted
c=0 # none as default
print(img_rgb.shape)
print(img_thermal.shape)
# print(img_shape)

combine =  addWeightedSmallImgToLargeImg(img_rgb,a,img_thermal,b,c,(0,0))

dir_file.mkdir("trans_fig")
dir_file.del_files("trans_fig")
saveFile = "trans_fig/trans_cv.jpg"  # 保存文件的路径
# cv2.imwrite(saveFile, img3, [int(cv2.IMWRITE_PNG_COMPRESSION), 8])  # 保存图像文件, 设置压缩比为 8
cv2.imwrite(saveFile, cat_img)  # 保存图像文件
saveFile = "trans_fig/combine.jpg"  # 保存文件的路径
cv2.imwrite(saveFile, combine)  # 保存图像文件

plt.imshow(cat_img)
plt.savefig("trans_fig/trans_plt.jpg")


# thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
# img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))