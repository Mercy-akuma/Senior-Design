import flir_image_extractor
import numpy as np
import pandas as pd
import os
import argparse

# 创建ArgumentParser对象，用于接收用户输入的参数
parser = argparse.ArgumentParser()

# 添加用户输入参数
parser.add_argument('--imagename', type=str, default='example', help='input the name of image')

# 解析用户输入参数
args = parser.parse_args()

def get_temperature(figure_path):
    # 读取红外图像对应的温度矩阵
    fir = flir_image_extractor.FlirImageExtractor()
    fir.process_image(figure_path)
    fir.plot()
    fir.save_images()
    # 提取温度数据
    t = fir.get_thermal_np()
    # 加载数据到pandas
    T = pd.DataFrame(t)
    # 存储为txt
    fn_prefix, _ = os.path.splitext(figure_path)
    txt_path = fn_prefix + ".txt"
    np.savetxt(txt_path ,T , fmt='%f',delimiter=',')
    print("Finish")

# 运行函数，将用户输入的imagename与路径拼接传入函数
get_temperature("Figure/" + args.imagename +".jpg")
