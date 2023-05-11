from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import os
import numpy as np
import threading

class RectangleDrawer:
    def __init__(self, image_path):
        # 打开图片并创建可编辑的图片对象
        self.image = Image.open(image_path)
        self.draw = ImageDraw.Draw(self.image)

        # 创建一个Tkinter窗口，并在其中显示图像
        self.root = Toplevel()
        self.canvas = Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor='nw')

        # 添加事件处理程序
        self.canvas.bind('<Button-1>', self.start_rect)
        self.canvas.bind('<B1-Motion>', self.draw_rect)
        self.canvas.bind('<ButtonRelease-1>', self.end_rect)

        # 初始化矩形变量
        self.rect_start = None
        self.rect_end = None
        self.rect_shape = None
        self.rect_coords = None

        self.root.protocol("WM_DELETE_WINDOW", self.quit_me)

        # 开始Tkinter主循环
        self.root.mainloop()

    def quit_me(self):
        self.root.quit()
        self.root.destroy()

    def start_rect(self, event):
        # 开始绘制矩形
        self.rect_start = (event.x, event.y)

    def draw_rect(self, event):
        # 绘制矩形
        if self.rect_shape:
            self.canvas.delete(self.rect_shape)
        self.rect_end = (event.x, event.y)
        self.rect_shape = self.canvas.create_rectangle(self.rect_start[0], self.rect_start[1], self.rect_end[0], self.rect_end[1], outline='red', width=3)

    def end_rect(self, event):
        # 矩形绘制完成，保存顶点坐标信息
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        small_y = min(y1, y2)
        big_y = max(y1, y2)
        small_x = min(x1, x2)
        big_x = max(x1, x2)
        self.rect_coords = np.array([small_x, small_y, big_x, big_y])

def find_input_box(optical_path, thermal_path):
    boxes = []
    rect_drawer = RectangleDrawer(optical_path)
    rect_coords = rect_drawer.rect_coords
    boxes.append(rect_coords)
    rect_drawer = RectangleDrawer(thermal_path)
    rect_coords = rect_drawer.rect_coords
    boxes.append(rect_coords)
    return boxes
    
# 测试代码
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    print(find_input_box('Figure/FLIR0359_rgbimage.jpg','Figure/FLIR0359_thermal.png'))



