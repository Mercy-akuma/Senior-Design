import os
import numpy as np
import cv2

imagename = "FPGA3"
from GUI import *
from arduino import *
import threading
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def run_gui():
    drawer = GUI()
    drawer.run()

def run_arduino():
    control_motor()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # print (os.getcwd())#获得当前目录

    # t1 = threading.Thread(target=run_gui)
    # t2 = threading.Thread(target=run_arduino)
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()

    run_gui()
