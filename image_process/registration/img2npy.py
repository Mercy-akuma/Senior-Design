
    # 尝试将转化的npy文件展示其中的内容。
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
# #####################  im 2 npy  #################

# path = r'../dataset1/jpgdata/test_jpg'
# out_path = r'../dataset1/luowei/'
def get_file(path,rule='.jpg'):
    all=[]
    for fpathe, dirs, files in os.walk(path):  # all 目录
        for f in files:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append((filename))
    return all

def img2npy(img):
    im1 = cv2.imread(img)
    im2=np.array(im1)
    # print(type(im2))

    np.save(img+'.npy',im2)
    print(tf.shape(im2))



 
def resize(img, size):
    # 先创建一个目标大小的幕布，然后将放缩好的图片贴到中央，这样就省去了两边填充留白的麻烦。
    canvas = Image.new("RGB", size=size, color="#7777")  
    
    target_width, target_height = size
    width, height = img.size
    offset_x = 0
    offset_y = 0
    if height > width:              # 高 是 长边
        height_ = target_height     # 直接将高调整为目标尺寸
        scale = height_ / height    # 计算高具体调整了多少，得出一个放缩比例
        width_ = int(width * scale) # 宽以相同的比例放缩
        offset_x = (target_width - width_) // 2     # 计算x方向单侧留白的距离
    else:   # 同上
        width_ = target_width
        scale = width_ / width
        height_ = int(height * scale)
        offset_y = (target_height - height_) // 2
 
    img = img.resize((width_, height_), Image.BILINEAR) # 将高和宽放缩
    canvas.paste(img, box=(offset_x, offset_y))         # 将放缩后的图片粘贴到幕布上
    # box参数用来确定要粘贴的图片左上角的位置。offset_x是x轴单侧留白，offset_y是y轴单侧留白，这样就能保证能将图片填充在幕布的中央
    
    return canvas
 
def resize_optical():
    img= Image.open('./ori_img/test_rgb_image.jpg')
    
    target__size=(480,360)  # 目标尺寸：宽为240，高为180
    res = resize(img,target__size)
    
    res.save('./ori_img/img_optical.jpg')

def resize_thermal():
    img= Image.open('./ori_img/test_thermal.jpg')
    
    target__size=(480,360)  # 目标尺寸：宽为240，高为180
    res = resize(img,target__size)
    
    res.save('./ori_img/img_thermal.jpg')
    
# def img2npy(): 
#     paths = get_file(r'D:\luowei_temp\SRCNN_try1\Super-Resolution_CNN-master\dataset1\jpgdata\test_jpg',rule='.jpg')
#     for ims in paths:
#         print(ims)
#         #cut path and '.jpg' ,保留 000000050755_bicLRx4图片名称 
#         file_name = ims.strip('D:\luowei_temp\SRCNN_try1\Super-Resolution_CNN-master\dataset1\jpgdata\test_jpg\'，.jpg') 
#         #可以打印出来看看cut以后的效果  
#         # print((file_name,type(file_name)))
        
#         im1 = cv2.imread(ims)
#         im2=np.array(im1)
#         # print(type(im2))

#         np.save(file_name+'.npy',im2)
#         print(tf.shape(im2))

# print("------show npy's shape_size---------")
# data = np.load('../dataset1/jpgdata/test_jpg/000000050755_bicLRx4.npy')
# print(data.shape)

def main():
    resize_optical()
    resize_thermal()
    paths = get_file('./ori_img/',rule='.jpg')
    for img in paths:
        img2npy(img)
        
    # paths = get_file(r'D:\luowei_temp\SRCNN_try1\Super-Resolution_CNN-master\dataset1\jpgdata\test_jpg',rule='.png')
    # for img in paths:
    #     img2npy(img)

    
if __name__ == "__main__":
    main()
    
    
