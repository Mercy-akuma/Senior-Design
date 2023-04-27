import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy.ma as npm
import torch
import torch.nn as nn
# import tensorflow as tf
from PIL import Image
import stat
from  matplotlib import patches
#这里是导入segment_anything包
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
sys.path.append("..")
from main import imagename

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        os.chmod(path,stat.S_IREAD|stat.S_IWRITE)
        print ("---  new folder...  ---")
        print ("---  OK  ---")
    else:
        print ("---  There is this folder!  ---")

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


class Sam():
    def __init__(self, 
                pth_name: str ="sam_vit_b_01ec64.pth",
                pth_type: str ="vit_b",
                log_file: str ="log.txt"
                
        ):
        # self.masked_optical="D:\_docker_mnt\py\sd\image_process\out_figs\sam_vit_h_4b8939.pth\img_optical.jpg+new.jpg"
        # self.masked_thermal="D:\_docker_mnt\py\sd\image_process\out_figs\sam_vit_h_4b8939.pth\img_thermal.jpg+new.jpg"
        self.masked_name=[]
        self.pth_name = pth_name
        self.pth_type = pth_type
        self.log_path = "./logs/"
        os.makedirs(self.log_path , exist_ok=True)
        self.log_file = self.log_path + log_file
        self.f=open(self.log_file,"w").close() # clear
        
        self.model_path = "./pth/" + self.pth_name
        start_time=time.time()
        self.logging("load model start at \t{}:\t {}".format(start_time,self.pth_name))
        self.sam = sam_model_registry[self.pth_type](checkpoint=self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = self.sam.to(self.device)
        end_time=time.time()
        self.logging("load model end at \t{}:\t {}".format(end_time,self.pth_name))
        self.logging("time spend \t{}:\t {}".format(end_time-start_time,self.pth_name))
        
    def logging(self,log:str):
        with open(self.log_file, 'a') as f:
            print(log, file=f)
    
        
    def gen_mask(self,image_path,output_folder,sam):
        current_time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("Model loaded done", current_time1)
        print("1111111111111111111111111111111111111111111111111111")
        #这里是加载图片
        image = cv2.imread(image_path)
        #输出图片加载完成的current时间
        current_time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("Image loaded done", current_time2)
        print("2222222222222222222222222222222222222222222222222222")
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        current_time3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("Predict done", current_time3)
        print("3333333333333333333333333333333333333333333333333333")
        print(masks[0])
        for i, mask in enumerate(masks):
            mask_array = mask['segmentation']
            mask_uint8 = (mask_array * 255).astype(np.uint8)

            # 为每个掩码生成一个唯一的文件名
            output_file = os.path.join(output_folder, f"mask_{i+1}.png")

            # 保存掩码
            cv2.imwrite(output_file, mask_uint8)

        height, width, _ = image.shape


        merged_mask = np.zeros((height, width), dtype=np.uint8)
        for i, mask in enumerate(masks):
            mask_array = mask['segmentation']
            mask_uint8 = (mask_array * 255).astype(np.uint8)

            # 为每个掩码生成一个唯一的文件名
            output_file = os.path.join(output_folder, f"mask_{i+1}.png")

            # 保存掩码
            cv2.imwrite(output_file, mask_uint8)

            merged_mask = np.maximum(merged_mask, mask_uint8)

        merged_output_file = os.path.join(output_folder, "mask_all.png")
        cv2.imwrite(merged_output_file, merged_mask)

    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    
    def show_points(self,coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                linewidth=1.25)
    
    
    def seg_point(self,img_path,out_path,point=[0,0]):
        predictor = SamPredictor(self.sam)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.set_image(image)
        
        input_point = np.array([point])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # 遍历读取每个扣出的结果
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca())
            self.show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            # plt.show()
            plt.savefig(out_path+"/seg_point_"+str(i)+".jpg")
            plt.close()    
            
    def seg_chip(self,img_path,out_img):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor = SamPredictor(self.sam)
        predictor.set_image(image)
        
        offset=30
        input_box = np.array([offset, offset, 240-offset, 180-offset])

        # input_point = np.array([[500, 375], [1125, 625]])
        # input_label = np.array([1, 0])

        # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask


        # input_point = np.array([point])
        # input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None, 
            box=input_box, 
            mask_input=None, 
            multimask_output=False, 
            return_logits=False
        )
        
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        self.show_mask(masks, plt.gca())
        # self.show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask , Score: {scores:.3f}", fontsize=18)
        plt.axis('off')
        # plt.show()
        plt.savefig(out_img)
        plt.close()  
        
        # new_image = np.ones(masks.shape)
        # new_image = new_image[masks]
        # print(new_image)
        # # new_image = npm.array(image, mask=masks)
        # cv2.imwrite(out_img+"+new.jpg", new_image*255)
        
        # for i, mask in enumerate(masks):
            # mask_array = mask['True']
        # print(masks[0])
        mask_uint8 = (masks[0] * 255).astype(np.uint8)
        # 保存掩码
        masked_name = out_img+"+new.jpg"
        self.masked_name.append(masked_name)
        cv2.imwrite(masked_name, mask_uint8)
        
        # mask_array = masks['segmentation']
        # mask_uint8 = (mask_array * 255).astype(np.uint8)
        # cv2.imwrite(out_img, mask_uint8)
        return masks[0]
        
           
    def run_sam(self,img_name,image_path):
        image = image_path + img_name
        output_folder = "./out_figs/"+self.pth_name+"/"
        output_path = output_folder + img_name
        os.makedirs(output_folder, exist_ok=True)

        # self.gen_mask(image_path,output_folder,self.sam)
        start_time=time.time()
        self.logging("run start at \t{}:\t {},{}".format(start_time,img_name,self.pth_name))
        # self.seg_point(image_path,output_path,[240,180])
        
        self.seg_chip(image,output_path)
        end_time=time.time()
        self.logging("run finish at \t{}:\t {},{}".format(end_time,img_name,self.pth_name))
        self.logging("time spend \t{}:\t {},{}".format(end_time-start_time,img_name,self.pth_name))

     
class Trans():
    def __init__(self, 
                img_optical: str = "./out_figs/img_optical.jpg",
                img_thermal: str = "./out_figs/reg_thermal.jpg",
        ):
        self.img_optical=img_optical
        self.img_thermal=img_thermal
      
        
        self.trans_optical="./out_figs/img.jpg"
        self.img_merged="../Figure/test_result.jpg"
        
        self.raw_optical= '../Figure/test_rgbimage.jpg'
        self.raw_thermal= '../Figure/test_thermal.png'
        self.cut_optical='./out_figs/cut_optical.jpg'
        self.reg_optical='./out_figs/reg_optical.jpg'
        
    def resize(self, img, size):
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
    
        img = img.resize((width_, height_), Image.Resampling.BILINEAR) # 将高和宽放缩
        canvas.paste(img, box=(offset_x, offset_y))         # 将放缩后的图片粘贴到幕布上
        # box参数用来确定要粘贴的图片左上角的位置。offset_x是x轴单侧留白，offset_y是y轴单侧留白，这样就能保证能将图片填充在幕布的中央
        
        return canvas
    
    def resize_input(self):
        target__size=(240,180)
        img_optical= Image.open(self.raw_optical)
        res = self.resize(img_optical,target__size)
        res.save(self.img_optical)
        img_thermal= Image.open(self.raw_thermal)
        res = self.resize(img_thermal,target__size)
        res.save(self.img_thermal)
        self.cut()
        reg_optical= Image.open(self.cut_optical)
        res = self.resize(reg_optical,target__size)
        res.save(self.reg_optical)
        
    def cut(self):
        offset_x=20 
        offset_y=30
        cut_optical= cv2.imread(self.img_optical)
        res = cut_optical[offset_y:180,offset_x:240-offset_x]
        cv2.imwrite(self.cut_optical,res) 
        
    def get_edge_box(self,mask_img):
        mask = cv2.imread(mask_img, cv2.COLOR_BGR2GRAY)
        # 转换成二值图
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        ax = plt.axes()
        # plt.imshow(mask, cmap='bone')
        # mask_find_bboxs
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask) # connectivity参数的默认值为8
        stats = stats[stats[:,4].argsort()]
        bboxs = stats[:-1]
        
        # print(cv2.__version__)

        # print(bboxs)
        b=bboxs[-1]
        # for b in bboxs:
        #     # rect = patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, edgecolor='r', facecolor='none')
        #     # print(b)
            
        pts3D=np.float32([[b[0], b[1]], [b[0]+b[2], b[1]], [b[0], b[1]+b[3]], [b[0]+b[2], b[1]+b[3]]])

        # plt.show()
        # cv2.imwrite('/out_figs/mask_bboxs.jpg',mask)

        # print(pts3D)
        return pts3D
        
    def perspective(self,img,pts3D1,pts3D2):
        # 计算透视放射矩阵
        M = cv2.getPerspectiveTransform(pts3D1, pts3D2)
        # 执行变换
        img = cv2.warpPerspective(img, M, (240,180))
        return img
    
    def reg(self,masked_optical,masked_thermal):
        pts3D1=self.get_edge_box(masked_optical)
        pts3D2=self.get_edge_box(masked_thermal)
        img_optical= cv2.imread(self.reg_optical)
        img=self.perspective(img_optical,pts3D1,pts3D2)
        cv2.imwrite(self.trans_optical,img)
        self.merge()

    def merge(self):
        img1=cv2.imread(self.trans_optical)
        img2=cv2.imread(self.img_thermal)
        res = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
        cv2.imwrite(self.img_merged,res)

    

def img_reg():
    os.chdir(os.path.dirname(__file__))
    trans=Trans()
    trans.resize_input()   
    
    # img_list=["test_rgb_image.jpg","test_thermal.png","test.jpg"]
    img_list=["reg_optical.jpg","reg_thermal.jpg"]
    image_path="./out_figs/"
    pth_list=["sam_vit_h_4b8939.pth","sam_vit_b_01ec64.pth","sam_vit_l_0b3195.pth"]
    pth_type=["vit_h","vit_b","vit_l"]
    
    skip=0
    for idx,pth_name in enumerate(pth_list):
        if idx != 1:
            continue
        log_file = pth_name + "_log.txt"
        sam=Sam(pth_name, pth_type[idx],log_file)
        if skip:
            break
        for img_name in img_list:
            sam.run_sam(img_name,image_path)
    
    trans.reg(sam.masked_name[0],sam.masked_name[1])
            
    
    
    
