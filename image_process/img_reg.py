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
                target_size: tuple = (640,480),
                pth_name: str ="sam_vit_b_01ec64.pth",
                pth_type: str ="vit_b",
                log_file: str ="log.txt",
                enable : bool = False
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
        if enable:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sam = self.sam.to(self.device)
        end_time=time.time()
        self.logging("load model end at \t{}:\t {}".format(end_time,self.pth_name))
        self.logging("time spend \t{}:\t {}".format(end_time-start_time,self.pth_name))

        self.target_size = target_size
        
    def logging(self,log:str):
        with open(self.log_file, 'a') as f:
            print(log, file=f)
    

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
            
    def seg_chip(self,img_path,out_img,box):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor = SamPredictor(self.sam)
        predictor.set_image(image)

        input_box = box

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
        
        self.seg_chip(image,output_path)
        end_time=time.time()
        self.logging("run finish at \t{}:\t {},{}".format(end_time,img_name,self.pth_name))
        self.logging("time spend \t{}:\t {},{}".format(end_time-start_time,img_name,self.pth_name))

    def run_sam_box(self,img_name,image_path,box):
        image = image_path + img_name
        output_folder = "./out_figs/"+self.pth_name+"/"
        output_path = output_folder + img_name
        os.makedirs(output_folder, exist_ok=True)

        # self.gen_mask(image_path,output_folder,self.sam)
        start_time=time.time()
        self.logging("run start at \t{}:\t {},{}".format(start_time,img_name,self.pth_name))
          
        self.seg_chip(image,output_path,box)
        end_time=time.time()
        self.logging("run finish at \t{}:\t {},{}".format(end_time,img_name,self.pth_name))
        self.logging("time spend \t{}:\t {},{}".format(end_time-start_time,img_name,self.pth_name))
     
class Trans():
    def __init__(self, 
                img_optical: str = "./out_figs/img_optical.jpg",
                img_thermal: str = "./out_figs/reg_thermal.jpg",
                raw_optical: str = '../Figure/test_rgbimage.jpg',
                raw_thermal: str ='../Figure/test_thermal.png',
                img_merged: str = "../Figure/test_result.jpg"

        ):
        self.img_optical=img_optical
        self.img_thermal=img_thermal
      
        
        self.trans_optical="./out_figs/img.jpg"
        self.img_merged = img_merged
        
        self.raw_optical= raw_optical
        self.raw_thermal= raw_thermal
        self.cut_optical='./out_figs/cut_optical.jpg'
        self.reg_optical='./out_figs/reg_optical.jpg'

        self.target_size = (640,480)
        
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
        
        # canvas=self.crop_img(canvas)
        return canvas
    
    def resize_input(self, target = (640,480)):
        self.target_size = target
        img_optical= Image.open(self.raw_optical)
        res = self.resize(img_optical,self.target_size)
        res.save(self.img_optical)
        img_thermal= Image.open(self.raw_thermal)
        res = self.resize(img_thermal,self.target_size)
        res.save(self.img_thermal)
        self.cut()
        reg_optical= Image.open(self.cut_optical)
        res = self.resize(reg_optical,self.target_size)
        res.save(self.reg_optical)
        
    def cut(self):
        res = cv2.imread(self.img_optical)
        cv2.imwrite(self.cut_optical,res)
    
    # ax = plt.axes()
        # # plt.imshow(mask, cmap='bone')
        # # mask_find_bboxs
        # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask) # connectivity参数的默认值为8
        # stats = stats[stats[:,4].argsort()]
        # bboxs = stats[:-1]    
        

    def resize_cv(self,img, new_size):
        height, width = img.shape[:2]
        new_width, new_height = new_size

        # determine scaling factor
        if height > width:
            scale = new_height / height
        else:
            scale = new_width / width

        # resize the image
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        return resized_img

    # def get_tight_boundary_box(self,max_contour):
    #     # Get bounding box of the contour
    #     x, y, w, h = cv2.boundingRect(max_contour)
        
    #     # Get mask for the contour
    #     mask = np.zeros(self.target_size, dtype=np.uint8)
    #     cv2.drawContours(mask, [max_contour], 0, 1, -1)
        
    #     # Check if 90% of the pixels within the bounding box are non-zero
    #     box_mask = mask[y:y+h, x:x+w]
    #     if cv2.countNonZero(box_mask) / float(box_mask.size) < 0.9:
    #         # Tighten the bounding box by iteratively removing 2 pixels from each edge
    #         for i in range(2):
    #             x, y, w, h = x + 2, y + 2, w - 4, h - 4
    #             box_mask = mask[y:y+h, x:x+w]
    #             if cv2.countNonZero(box_mask) / float(box_mask.size) >= 0.9:
    #                 break
        
    #     return x, y, w, h
    
    def get_tight_boundary_box(self, max_contour):
        x, y, w, h = cv2.boundingRect(max_contour)
        mask = np.zeros(self.target_size, dtype=np.uint8)
        cv2.drawContours(mask, [max_contour], 0, 255, thickness=cv2.FILLED)

        W,H= w, h
        factor = 0.05
        # check rows
        roi = mask[y:y+h, x:x+w]
        reversed_rows_roi = np.flipud(roi)
        for row in roi:
            non_zero_count = np.count_nonzero(row)
            if non_zero_count <= W*factor:
                # Reduce the height of the bounding box by 1 pixel
                y += 1
                h -= 1
            else:    
                break
            
        for row in reversed_rows_roi:
            non_zero_count = np.count_nonzero(row)
            if non_zero_count <= W*factor:
                # Reduce the height of the bounding box by 1 pixel
                h -= 1
            else:    
                break
        
        roi_T=roi.T
        reversed_cols_roi = np.flipud(roi_T)
        
        # check cols
        for col in roi_T:
            non_zero_count = np.count_nonzero(col)
            if non_zero_count <= H*factor:
                # Reduce the height of the bounding box by 1 pixel
                x += 1
                w -= 1
            else:    
                break
            
        for col in reversed_cols_roi:
            non_zero_count = np.count_nonzero(col)
            if non_zero_count <= H*factor:
                # Reduce the height of the bounding box by 1 pixel
                w -= 1
            else:    
                break

        return x, y, w, h


    def get_edge_box(self,mask_img):
        mask = cv2.imread(mask_img, cv2.COLOR_BGR2GRAY)
        # 转换成二值图
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # boxes = []
        # area = []
        # # dot=[]
        # for idx,c in enumerate(contours):
        #     area.append((cv2.contourArea(c),idx))

        contours_list=[(c,cv2.contourArea(c)) for c in contours]
        # print(contours_list)
        # c = contours[idx]
        max_contour=max(contours_list, key=lambda x: x[1])[0]
        # boxes=cv2.minAreaRect(max_contour)
        x , y, w, h=cv2.boundingRect(max_contour)
        print(x , y, w, h)
        
        x , y, w, h=self.get_tight_boundary_box(max_contour)
        print(x , y, w, h)
        pts3D=np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        
        # pts3D = np.int0(pts3D)
        # # pts3D = np.float32(pts3D)
        
        # new_image=cv2.drawContours(mask, [pts3D], 0, (0, 255, 0), 2)
        # cv2.imwrite
        pt1 = (int(pts3D[0][0]), int(pts3D[0][1]))
        pt2 = (int(pts3D[3][0]), int(pts3D[3][1]))

        # Draw the rectangle on the image
        color = (0, 0, 255)
        thickness = 2
        basename = mask_img.split("+")[0]
        img = cv2.imread(basename)
        # pil_img= Image.open(basename)
        img=self.crop_img(img)
        img=self.resize_cv(img,self.target_size)
        
        # # Crop the canvas to remove any remaining whitespace
        # bbox = canvas.getbbox()
        # cropped_canvas = canvas.crop(bbox)

        # # Convert the cropped canvas to a NumPy array
        # np_image = np.asarray(cropped_canvas)

        # Convert the NumPy array to a cv2 format image
        # img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        # img=self.crop_img(img)
        
        

        cv2.circle(img, (640,480), 5, (0, 0, 255), -1)
        
        
        cv2.rectangle(img, pt1, pt2, color, thickness)
        cv2.imwrite(mask_img+"box.jpg", img)
        
        # pts3D = np.float32(pts3D)
        # pts3D = np.float32([[b[0], b[1]], [b[0]+b[2], b[1]], [b[0], b[1]+b[3]], [b[0]+b[2], b[1]+b[3]]])

    
            # # 找到边界坐标
            # min_list=[] # 保存单个轮廓的信息，x,y,w,h,area。 x,y 为起始点坐标
            # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            # min_list.append(x)
            # min_list.append(y)
            # min_list.append(w)
            # min_list.append(h)
            # min_list.append(w*h) # 把轮廓面积也添加到 dot 中
            # dot.append(min_list)

        # # 找出最大矩形的 x,y,w,h,area
        # max_area=dot[0][4] # 把第一个矩形面积当作最大矩形面积
        # for inlist in dot:
        #     area=inlist[4]
        #     if area >= max_area:
        #         x=inlist[0]
        #         y=inlist[1]
        #         w=inlist[2]
        #         h=inlist[3]
        #         max_area=area

        # print(x,y,w,h)
        # pts3D=np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        
        # for bbox in bounding_boxes:
        #     [x , y, w, h] = bbox
        #     cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # print(cv2.__version__)
        # areas = []
 
        # for c in range(len(contours)):
        #     areas.append(cv2.contourArea(contours[c]))
        # max_id = areas.index(max(areas))
        # max_rect = cv2.minAreaRect(contours[max_id])

        
        # max_box = cv2.boxPoints(max_rect)
        # max_box = np.int0(max_box)
        # pts3D=np.float32(max_box)

        # [[ 97. 119.]
        # [478. 119.]
        # [ 97. 405.]
        # [478. 405.]]

        # [[ 95. 401.]
        # [ 96. 117.]
        # [497. 120.]
        # [ 96. 404.]]

        # [[ 53. 426.]
        # [ 55.  47.]
        # [567.  51.]
        # [564. 430.]]

        # print(bboxs)
        # b= cv2.boundingRect(contours[-1])
        # print(b)
        # for b in bboxs:
        #     # rect = patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, edgecolor='r', facecolor='none')
        #     # print(b)
        
        # 4 points 00 10 01 11
        # pts3D=np.float32([[b[0], b[1]], [b[0]+b[2], b[1]], [b[0], b[1]+b[3]], [b[0]+b[2], b[1]+b[3]]])
        return pts3D
        
    def perspective(self,img,pts3D1,pts3D2):
        # 计算透视放射矩阵
        M = cv2.getPerspectiveTransform(pts3D1, pts3D2)
        # 执行变换
        img = cv2.warpPerspective(img, M, self.target_size)
        return img
    
    def reg(self,masked_optical,masked_thermal):
        pts3D1=self.get_edge_box(masked_optical)
        pts3D2=self.get_edge_box(masked_thermal)
        
        # cut
        col_width = pts3D1[1][0] - pts3D1[0][0]
        row_width = pts3D1[2][1] - pts3D1[0][1]

        map_col_width = pts3D2[1][0] -pts3D2[0][0]
        map_row_width = pts3D2[2][1] -pts3D2[0][1]

        col_width = int(map_col_width * row_width / map_row_width)
        # set 
        pts3D1[1][0] = pts3D1[0][0] + col_width
        pts3D1[3][0] = pts3D1[1][0]

        # print(pts3D1)
        # print(pts3D2)

        img_optical= cv2.imread(self.reg_optical)
        img=self.perspective(img_optical,pts3D1,pts3D2)
        cv2.imwrite(self.trans_optical,img)
        self.merge()

    def merge(self):
        img1=cv2.imread(self.trans_optical)
        img2=cv2.imread(self.img_thermal)
        res = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
        cv2.imwrite(self.img_merged,res)

    def crop_img(self,img):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to obtain a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image using the bounding box coordinates
        img_cropped = img[y:y+h, x:x+w]
        
        return img_cropped

class Pre_process():
    def __init__(self, 
                imagename : str= "FPGA1",
                target_size: tuple = (640,480)
        ):
        self.imagename = imagename
        self.target_size = target_size
        # os.chdir("./image_process")
        os.chdir(os.path.dirname(__file__))
        # img_list=["test_rgb_image.jpg","test_thermal.png","test.jpg"]
        self.img_list=["reg_optical.jpg","reg_thermal.jpg"]
        self.image_path="./out_figs/"
        self.pth_list=["sam_vit_h_4b8939.pth","sam_vit_b_01ec64.pth","sam_vit_l_0b3195.pth"]
        self.pth_type=["vit_h","vit_b","vit_l"]
        self.idx=1
        self.pth_name=self.pth_list[self.idx]
        
        self.log_file = self.pth_name + "_log.txt"
        self.sam=Sam(self.target_size,self.pth_name, self.pth_type[self.idx],self.log_file)

        # resize images
        prefix = "../Figure/" + self.imagename
        self.trans=Trans(raw_optical= prefix + "_rgbimage.jpg", raw_thermal=prefix + "_thermal.png", img_merged= prefix + "_result.jpg")
        self.trans.resize_input()
        # Find inputbox here:


    def img_reg(self,boxes):

        # box = np.array([ small_y,  small_x,big_y, big_x])
        # modify img_list for new pair of image TODO
        # for img_name in img_list:
        #     self.sam.run_sam(img_name,image_path)
        for idx,img_name in enumerate(self.img_list):
            self.sam.run_sam_box(img_name,self.image_path,boxes[idx])

        self.trans.reg(self.sam.masked_name[0],self.sam.masked_name[1])    

    
    
    
