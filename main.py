import os
import numpy as np
import cv2
from GUI import *
from arduino import *
import threading
from image_process.img_reg import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

imagename = "test"

def run_gui():
    drawer = GUI()
    drawer.run()

def run_arduino():
    control_motor()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # print (os.getcwd())#获得当前目录
    # img_reg()
    t1 = threading.Thread(target=run_gui)
    # t2 = threading.Thread(target=run_arduino)
    t1.start()
    # t2.start()
    t1.join()
    # t2.join()


    # # Creating a contour plot
    # data = np.loadtxt("Figure/test.txt", delimiter=',')
    # # Finding the index of the maximum value in the data array
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # # Plotting the contour plot
    # c = ax.contour(data, colors='black')

    # # Adding labels for contour lines
    # ax.clabel(c, inline=True, fontsize=10)

    # # Setting the labels for the axes
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')

    # # Inverting the y-axis to move the origin to the top left corner
    # ax.invert_yaxis()

    # # Save the figure to the Figure folder
    # fig.savefig("Figure/fig.png")





