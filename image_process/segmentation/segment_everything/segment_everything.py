# from segment_anything import build_sam, SamAutomaticMaskGenerator
# import numpy as np
# # import torch
# import matplotlib.pyplot as plt
# import cv2


# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

# image="./ori_figs/test_rgb_image.jpg"

# my_pth="/pth/sam_vit_h_4b8939.pth"

# mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=my_pth))
# masks = mask_generator.generate(image)

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 




#这是一个seganythng的demo
#先导入必要的包
import os
import sys
import time
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


#这里是导入segment_anything包
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry


def logging(filename,log):
    f= open(filename, 'a' )
    print(log, file=f)
    f.close()


def gen_mask(image_path,output_folder,sam):
    
    

    #输出模型加载完成的current时间

    current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Model loaded done", current_time1)
    print("1111111111111111111111111111111111111111111111111111")
    #这里是加载图片
    image = cv2.imread(image_path)
    #输出图片加载完成的current时间
    current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Image loaded done", current_time2)
    print("2222222222222222222222222222222222222222222222222222")

    #这里是加载图片，这里的image_path是图片的路径

    #这里是预测,不用提示词,进行全图分割
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    #使用提示词,进行局部分割
    # predictor = SamPredictor(sam)
    # predictor.set_image(image)
    # masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False)

    #输出预测完成的current时间
    current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Predict done", current_time3)
    print("3333333333333333333333333333333333333333333333333333")

    # #展示预测结果img和mask
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(masks[0])
    # plt.show()


    #保存mask
    print(masks[0])


    # # mask_array = masks[0]['segmentation']


    # # mask_uint8 = (mask_array * 255).astype(np.uint8)

    # cv2.imwrite(output_path, mask_uint8)

    #循环保存mask
    # 遍历 masks 列表并保存每个掩码
    for i, mask in enumerate(masks):
        mask_array = mask['segmentation']
        mask_uint8 = (mask_array * 255).astype(np.uint8)

        # 为每个掩码生成一个唯一的文件名
        output_file = os.path.join(output_folder, f"mask_{i+1}.png")

        # 保存掩码
        cv2.imwrite(output_file, mask_uint8)


    #输出完整的mask
    # 获取输入图像的尺寸
    height, width, _ = image.shape

    # 创建一个全零数组，用于合并掩码
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历 masks 列表并合并每个掩码
    for i, mask in enumerate(masks):
        mask_array = mask['segmentation']
        mask_uint8 = (mask_array * 255).astype(np.uint8)

        # 为每个掩码生成一个唯一的文件名
        output_file = os.path.join(output_folder, f"mask_{i+1}.png")

        # 保存掩码
        cv2.imwrite(output_file, mask_uint8)

        # 将当前掩码添加到合并掩码上
        merged_mask = np.maximum(merged_mask, mask_uint8)

    # 保存合并后的掩码
    merged_output_file = os.path.join(output_folder, "mask_all.png")
    cv2.imwrite(merged_output_file, merged_mask)
    
    
def main():    
    # #释放cv2
    # cv2.destroyAllWindows()

    #输入必要的参数

    # image="./ori_figs/test_rgb_image.jpg"

    # my_pth="/pth/sam_vit_h_4b8939.pth"


   

    #这里是加载模型，这里的model_path是模型的路径，sam_model_registry是模型的名称

    img_list=["test_rgb_image.jpg","test_thermal.png","test.jpg"]
    pth_list=["sam_vit_h_4b8939.pth","sam_vit_b_01ec64.pth","sam_vit_l_0b3195.pth"]
    pth_type=["vit_h","vit_b","vit_l"]
    
    
    idx=-1
    for pth_name in pth_list:
        idx+=1
        for img_name in img_list:
            image_path = "./ori_figs/" + img_name
            output_folder = "./out_figs/"+pth_name+"/" + img_name
            model_path = "./pth/" +pth_name
            # 确保输出文件夹存在
            os.makedirs(output_folder, exist_ok=True)
            print("start: {},{}".format(img_name,pth_name))
            #官方demo加载模型的方式
            sam = sam_model_registry[pth_type[idx]](checkpoint=model_path)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # sam = sam.to(device)
            gen_mask(image_path,output_folder,sam)
            
            print("finish: {},{}".format(img_name,pth_name))
            # logging("log.txt","{},{}".format(img_name,pth_name))
        

if __name__ == "__main__":
    main()
    
    
