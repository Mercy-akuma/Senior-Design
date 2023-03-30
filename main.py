#coding=utf-8
import subprocess
import os
import numpy as np
import cv2

imagename = "test"

def cmd_enable():
    # cmd = 'cmd.exe d:/start.bat'
    p = subprocess.Popen("cmd.exe /c" + "EnableUSB.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    curline = p.stdout.readline()
    while (curline != b''):
        print(curline)
        curline = p.stdout.readline()
    p.wait()
    print(p.returncode)

def cmd_disable():
    # cmd = 'cmd.exe d:/start.bat'
    p = subprocess.Popen("cmd.exe /c" + "DisableUSB.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    curline = p.stdout.readline()
    while (curline != b''):
        print(curline)
        curline = p.stdout.readline()
    p.wait()
    print(p.returncode)

def gui(figure_path, txt_path):
    img = cv2.imread(figure_path)     #读取图片
    a = []
    b = []
    temperature_data = np.loadtxt(txt_path, delimiter=',')
    print("The shape of this image is: ", temperature_data.shape)
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            a.append(x)
            b.append(y)
            # cv2.circle(img, (x, y), 3, (255, 0, 0), thickness=-1)
            cv2.imshow("image", img)
            print("Location: [{},{}], Temperature: {} °C".format(a[-1], b[-1], temperature_data[b[-1]][a[-1]]))	#输出最后一次点击的坐标
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # print (os.getcwd())#获得当前目录
    # cmd_enable()
    gui("Figure/" + imagename + "_thermal.png", "Figure/" + imagename + ".txt")

    # img1 = cv2.imread('Figure/test_thermal.png')
    # img2 = cv2.imread('Figure/test_rgb_image.jpg')
    # combine = cv2.addWeighted(img1,0.5,cv2.resize(img2,(240,180)),0.5,0)
    # cv2.imshow('combine',combine)
    # cv2.waitKey(0)