import flir_image_extractor
import numpy as np
import pandas as pd
import os
from main import imagename

def get_temperature(figure_path):
    # 读取红外图像对应的温度矩阵
    fir = flir_image_extractor.FlirImageExtractor()
    fir.process_image(figure_path)
    fir.plot()
    fir.save_images()
    # #提取温度数据
    t = fir.get_thermal_np()
    # 加载数据到pandas
    T = pd.DataFrame(t)
    # 存储为txt
    fn_prefix, _ = os.path.splitext(figure_path)
    txt_path = fn_prefix + ".txt"
    np.savetxt(txt_path ,T , fmt='%f',delimiter=',')
    print("Finish")

get_temperature("Figure/" + imagename +".jpg")
