import os
import numpy as np
import cv2
from GUI import *
from arduino import *
import threading
from image_process.img_reg import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

imagename = "test"

def run_gui():
    drawer = GUI()
    drawer.run()
    
    # Plot the predicted values against the actual values
    plt.figure()
    plt.scatter(drawer.y, drawer.y_pred)
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title('Linear Regression Results')
    # Save the linear regression plot as an image file
    plt.savefig('Figure/linear_regression_plot.png')

    # Calculate the R-squared value
    r_squared = drawer.reg.score(drawer.X, drawer.y)
    logging("R-squared value: {} ".format(r_squared))
    logging("reg.coef_ value: {} ".format(drawer.reg.coef_))
    logging("reg.intercept_ value: {} ".format(drawer.reg.intercept_))
    logging("b: {} ".format(min(drawer.y)))


    plot_mask(drawer)

def logging(log:str):
    with open("main_log.txt", 'a') as f:
        print(log, file=f)
    
    

def plot_mask(drawer):
    mask=drawer.mask
    rectangle=drawer.plt_rectangle
    f=open("mask.txt","w")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            f.write("{:.3f} ".format(mask[i][j]))
        f.write("\n")  
    f.close()  
    

    # Plot the predicted values against the actual values
    img=plt.figure()
    ax=img.add_subplot(111)
    plt.imshow(mask)
    # contour=plt.contourf(mask,np.arange(0.0, 1.1, 0.1))
    contour=plt.contourf(mask)
    # plt.colorbar(img)
    plt.colorbar(contour)
    # plt.clabel(contour,fontsize=10,inline=1, colors='k')
    
    currentAxis = img.gca()
    currentAxis.add_patch(rectangle)
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    
    plt.savefig('Figure/mask.png')


def run_arduino():
    control_motor()

# python main.py > main.txt
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # print (os.getcwd())#获得当前目录
    # img_reg()
    
    open("main_log.txt", 'w').close()#clear
    
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





