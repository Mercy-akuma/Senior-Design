import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import subprocess
import threading
from sklearn.linear_model import LinearRegression
import serial
import matplotlib.pyplot as plt
import math
import glob
import os
import os.path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

from scipy.optimize import nnls
# from  main import key

class GUI:
    def __init__(self):
        self.rectangles = []
        self.total_power = None # Initialize total_power to None
        # parameters when drawing a rectangle
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None
        self.imagename = None
        self.folderpath = None
        self.mask = None
        self.rect = None
        self.history = []
        self.image = None
        self.data = None # temperature data
        self.image_path = None
        self.image_width = None
        self.image_height = None
        # self.key = None
        self.rect_count = 0 # Initialize a counter for the rectangles
        self.root = None
        self.output_window = None
        self.flag = 0

    def del_files(self, path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            if os.path.isdir(f_path):
                self.del_files(f_path)
            elif os.path.basename(f_path) != "rectangles.dat": # Check if the file is not 'rectangles.dat'
                os.remove(f_path)

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

    def open_image(self, image_path=None):
        self.rectangles = []
        self.total_power = None # Initialize total_power to None
        # parameters when drawing a rectangle
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None
        self.imagename = None
        self.folderpath = None
        self.mask = None
        self.rect = None
        self.image = None
        self.data = None # temperature data
        self.image_path = None
        self.image_width = None
        self.image_height = None
        self.flag = 1
        # self.key = None
        self.rect_count = 0 # Initialize a counter for the rectangles
        self.canvas.delete(tk.ALL)
        if image_path:
            self.image_path = image_path.replace("\\", "/")
            
        else:
            self.image_path = filedialog.askopenfilename(filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])

        directory_path = '_'.join(self.image_path.split('_')[:-1])
        self.imagename = directory_path.split('/')[-1]
        self.folderpath = os.path.dirname(self.image_path)

        file_path = directory_path + '.txt'
        self.data = np.loadtxt(file_path, delimiter=',')

        image = Image.open(self.image_path)
        image = self.resize(image, (640,480))
        self.image_width, self.image_height = image.size
        # self.image = ImageTk.PhotoImage(file=self.image_path)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.config(width=self.image_width, height=self.image_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # check if the directory exists
        output_directory = os.path.join(os.path.dirname(self.image_path), self.imagename + "_analysis")
        if not os.path.exists(output_directory):
            # if directory does not exist, create it
            os.mkdir(output_directory)
        self.del_files(output_directory)
        open(self.folderpath + "/" + self.imagename + "_analysis" + "/image_log.txt", 'w').close()
        open(self.folderpath + "/" + self.imagename + "_analysis" + "/result.txt", 'w').close()

    # def open_text(self):
    #     self.data = None
    #     file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    #     self.data = np.transpose(np.loadtxt(file_path, delimiter=','))
    
    def start_rect(self, event):
        if self.flag == 0:
            return
        self.rect_start_x = self.canvas.canvasx(event.x)
        self.rect_start_y = self.canvas.canvasy(event.y)

    def draw_rect(self, event):
        if self.flag == 0:
            return
        self.rect_end_x = self.canvas.canvasx(event.x)
        self.rect_end_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect_start_x = max(0, min(self.rect_start_x, self.image_width))
        self.rect_start_y = max(0, min(self.rect_start_y, self.image_height))
        self.rect_end_x = max(0, min(self.rect_end_x, self.image_width))
        self.rect_end_y = max(0, min(self.rect_end_y, self.image_height))
        # Save the rectangle as an object on the canvas
        self.rect = self.canvas.create_rectangle(self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y, outline='red')

    def save_location(self):
        """
        This function saves the self.rectangles list to a binary file in the output directory.
        """
        if self.flag == 0:
            return
        output_directory = os.path.join(os.path.dirname(self.image_path), self.imagename + "_analysis")
        with open(os.path.join(output_directory, "rectangles.dat"), "wb") as f:
            pickle.dump(self.rectangles, f)
        print("Saved successfully")

    def load_location(self):
        """
        This function loads the rectangles from a binary file and adds them to self.rectangles.
        """
        if self.flag == 0:
            return
        output_directory = os.path.join(os.path.dirname(self.image_path), self.imagename + "_analysis")
        with open(os.path.join(output_directory, "rectangles.dat"), "rb") as f:
            tmp_rectangles = pickle.load(f)
        print("Loaded successfully")

        for old_rect in self.history:
            self.canvas.delete(old_rect[0])
            self.canvas.delete(old_rect[1])

        orig_height = self.data.shape[0]
        orig_width = self.data.shape[1]
        trans_height = self.image_height
        trans_width = self.image_width
        count = 0
        for rectangle in tmp_rectangles:
            count += 1
            start_x = rectangle[0] / (orig_width/trans_width)
            start_y = rectangle[1] / (orig_height/trans_height)
            end_x = rectangle[2] / (orig_width/trans_width)
            end_y = rectangle[3] / (orig_height/trans_height)
            self.rect_start_x = start_x
            self.rect_start_y = start_y
            self.rect_end_x = end_x
            self.rect_end_y = end_y
            self.save_rect(0)
            self.rect = self.canvas.create_rectangle(start_x, start_y, end_x, end_y, outline='red')
            self.text = self.canvas.create_text((start_x + end_x)/2, (start_y + end_y)/2, text=str(count), fill='blue', font=("Purisa", 30))
            self.history.append((self.rect, self.text))
        self.rect = None
        self.rect_count = len(self.rectangles)
        
        # self.rectangles= tmp_rectangles

    def save_rect(self, event):
        if self.flag == 0:
            return
        # Create a mask for the rectangle
        orig_height = self.data.shape[0]
        orig_width = self.data.shape[1]
        trans_height = self.image_height
        trans_width = self.image_width
        
        # print([self.rect_start_y,self.rect_end_y,self.rect_start_x,self.rect_end_x])
        
        # Add a label with the ordinal number of the rectangle in the middle of the rectangle
        self.text = self.canvas.create_text((self.rect_start_x + self.rect_end_x)/2, (self.rect_start_y + self.rect_end_y)/2, text=str(self.rect_count + 1), fill='blue', font=("Purisa", 30))
        self.history.append((self.rect, self.text))
        self.rect_start_y = self.rect_start_y * (orig_height/trans_height)
        self.rect_end_y = self.rect_end_y * (orig_height/trans_height)
        self.rect_start_x = self.rect_start_x * (orig_width/trans_width)
        self.rect_end_x = self.rect_end_x * (orig_width/trans_width)
        self.mask = np.zeros((orig_height, orig_width))
        
        # print([orig_height,trans_height,orig_width,trans_width])
        # print([self.rect_start_y,self.rect_end_y,self.rect_start_x,self.rect_end_x])
        big_x = int(max(self.rect_end_x,self.rect_start_x))
        small_x= int(min(self.rect_end_x,self.rect_start_x))
        big_y = int(max(self.rect_end_y,self.rect_start_y))
        small_y = int(min(self.rect_end_y,self.rect_start_y))
        
        self.mask[small_y:big_y, small_x:big_x] = 1

        w = big_x - small_x
        l = big_y - small_y
        plt_rect = plt.Rectangle(xy=(small_x,small_y),width=w, height=l,linewidth=1, edgecolor='r',facecolor='r')

        area = w*l
        # print("w,l: {},{}".format(w,l))
        c = abs(2*(w+l))
        radius = np.sqrt(w**2+l**2)/2
        for i in range(orig_height):
            for j in range(orig_width):
                if self.mask[i][j] == 0:
                    # y -- i, x -- j
                    location = [i-big_y,i-small_y,j-big_x,j-small_x]
                    if location[1] < 0 and location[3] < 0:
                        r = np.sqrt(location[1]**2 + location[3]**2 )
                    elif location[0] > 0 and location[2] > 0:
                        r = np.sqrt(location[0]**2 + location[2]**2 )
                    elif location[1] < 0 and location[2] > 0 :
                        r = np.sqrt(location[1]**2 + location[2]**2 )
                    elif location[0] > 0 and location[3] < 0:
                        r = np.sqrt(location[0]**2 + location[3]**2 )
                    elif location[0] <= 0 and location[1] >= 0 and location[2] > 0:
                        r = abs(location[2])
                    elif location[0] <= 0 and location[1] >= 0 and location[3] < 0:
                        r = abs(location[3])
                    elif location[0] > 0 and location[2] <= 0 and location[3] >= 0:
                        r = abs(location[0])
                    elif location[1] < 0 and location[2] <= 0 and location[3] >= 0:
                        r = abs(location[1])
                    else:
                        continue # not possible, same with self.mask[i][j] == 1
                    new_area = area + 2*r*(w+l) + math.pi*r**2
                    new_c = c + 2 * math.pi * r
                    # self.mask[i][j] = 1/r
                    # self.mask[i][j] = area/new_area
                    self.mask[i][j] = c/new_c
                    # self.mask[i][j] = 1/r

        self.rectangles.append((small_x, small_y, big_x, big_y, self.mask))
        self.rect = None
        self.rect_count += 1 # Increment the counter for each new rectangle draw
        
    def output_power(self):
        # Linear regression to get the energy of each rectangle
        X = []
        for rect in self.rectangles:
            small_x, small_y, big_x, big_y, mask = rect
            X.append(mask.flatten())
            
        X = np.array(X)
        X = X.transpose()
        y = self.data.flatten()
        
        # linear regression
        reg = LinearRegression(positive = True).fit(X, y)
        rectangle_energy = reg.coef_
        rectangle_energy = rectangle_energy / np.sum(rectangle_energy) * self.total_power
        y_pred = reg.predict(X)

        # Plot the predicted values against the actual values
        plt.figure()
        plt.scatter(y, y_pred)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Linear Regression Results')
        # Save the linear regression plot as an image file
        plt.savefig(self.folderpath + "/" + self.imagename + "_analysis" + "/linear_regression_plot.png")

        # Calculate the R-squared value
        r_squared = reg.score(X, y)
        print("R-squared value:", r_squared)
        self.logging("R-squared value: {} ".format(r_squared))
        self.logging("reg.coef_ value: {} ".format(reg.coef_))
        self.logging("reg.intercept_ value: {} ".format(reg.intercept_))
        self.logging("b: {} ".format(min(y)))

        output_text = ""
        for i in range(len(rectangle_energy)):
            output_text += "Component {}'s power consumption is {}\n".format(i + 1, rectangle_energy[i])
            self.plot_mask(i)
        self.out_result(output_text)
        self.output_window = tk.Toplevel()
        self.output_window.geometry("500x500") # increase the size of the window

        self.output_window.title("Power Consumption Output")
        self.output_text = tk.Text(self.output_window, height=10, width=100)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.output_text.insert(tk.END, output_text)
        self.scrollbar = tk.Scrollbar(self.output_window)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar.config(command=self.output_text.yview)
        self.output_text.config(yscrollcommand=self.scrollbar.set)
        self.output_window.protocol("WM_DELETE_WINDOW", self.quit_window)
        self.output_window.mainloop()

    def logging(self, log:str):
        with open(self.folderpath + "/" + self.imagename + "_analysis" + "/image_log.txt", 'a') as f:
            print(log, file=f)

    def out_result(self, log:str):
        with open(self.folderpath + "/" + self.imagename + "_analysis" + "/result.txt", 'a') as f:
            print(log, file=f)

    def plot_mask(self, i:int): # i means the index of maskimage.png
        # "Component {}'s mask".fomat(i)
        small_x, small_y, big_x, big_y, mask = self.rectangles[i]
        w = big_x - small_x
        l = big_y - small_y
        rectangle = plt.Rectangle(xy=(small_x,small_y),width=w, height=l,linewidth=1, edgecolor='r',facecolor='r')

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
        
        plt.savefig(self.folderpath + "/" + self.imagename + "_analysis" + "/mask_{}.png".format(i + 1))

    def show_image(self):
        # Create a new window
        image_window = tk.Toplevel()
        image_window.title("Image Viewer")
        
        # Load the image
        image_path = filedialog.askopenfilename(filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        image = Image.open(image_path)
        
        # Resize the image to fit the window
        if image.width < 300 or image.height < 200:
            image = image.resize((image.width*2, image.height*2), Image.ANTIALIAS)
        
        # Create a PhotoImage object from the image
        photo = ImageTk.PhotoImage(image)
        
        # Add the image to a Label widget
        label = tk.Label(image_window, image=photo)
        label.image = photo # Keep a reference to the PhotoImage object to prevent it from being garbage collected
        
        # Pack the Label widget
        label.pack()
        
    def cmd_enable(self):
        # cmd = 'cmd.exe d:/start.bat'
        p = subprocess.Popen("cmd.exe /c" + "EnableUSB.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        curline = p.stdout.readline()
        while (curline != b''):
            print(curline)
            curline = p.stdout.readline()
        p.wait()

    def cmd_disable(self):
        # cmd = 'cmd.exe d:/start.bat'
        p = subprocess.Popen("cmd.exe /c" + "DisableUSB.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        curline = p.stdout.readline()
        while (curline != b''):
            print(curline)
            curline = p.stdout.readline()
        p.wait()
    
    def show_pixel_position(self, event):
            x, y = event.x, event.y
            print("Pixel position: ({}, {})".format(x, y))
    

    # def check_mew_image(self):
    #     # 检测文件夹下最新的jpg文件
    #     folder_path = "Figure/batch_process"
    #     files = glob.glob(os.path.join(folder_path, '*.jpg')) # 查找所有jpg文件
    #     if len(files) == 0:
    #         print("No jpg file found")
    #         return None
    #     else:
    #         latest_file = max(files, key=os.path.getctime) # 找到最新的文件
    #         print("Latest jpg file found:", latest_file)
    #     self.open_image(latest_file)

    def show_contour(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plotting the contour plot
        c = ax.contour(self.data, colors='black')
        # Adding labels for contour lines
        ax.clabel(c, inline=True, fontsize=10)
        # Setting the labels for the axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Inverting the y-axis to move the origin to the top left corner
        ax.invert_yaxis()
        # Save the figure to the Figure folder
        fig.savefig(self.folderpath + "/" + self.imagename + "_analysis"+ "/contour.png")
        # Show the figure in a new window
        img = Image.open(self.folderpath + "/" + self.imagename + "_analysis" + "/contour.png")
        img.show()
    
    def quit_me(self):
        print('quit')
        self.root.quit()
        self.root.destroy()

    def quit_window(self):
        self.output_window.quit()
        self.output_window.destroy()

    def run(self):
        self.root = tk.Tk()
        self.root.title("Power Consumption GUI")
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Add a label and entry for total_power
        tk.Label(self.root, text="Total Power:").pack(side=tk.LEFT)
        total_power_entry = tk.Entry(self.root)
        total_power_entry.pack(side=tk.LEFT)
        # Add a button to set total_power
        set_power_button = tk.Button(self.root, text="Set Power", command=lambda: setattr(self, 'total_power', float(total_power_entry.get())))
        set_power_button.pack(side=tk.LEFT)

        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        operation_menu = tk.Menu(menu_bar, tearoff=0)
        # File_menu
        file_menu.add_command(label="Import Image", command=self.open_image)
        # file_menu.add_command(label="Import Text", command=self.open_text)
        file_menu.add_command(label="Show Image", command=self.show_image)
        # file_menu.add_command(label="Check New Image", command=self.check_mew_image)
        # Operation_menu
        # operation_menu.add_command(label="Output Power", command=self.output_power)
        operation_menu.add_command(label="Enable Connection", command=self.cmd_enable)
        operation_menu.add_command(label="Disable Connection", command=self.cmd_disable)
        # operation_menu.add_command(label="Enable Drawing", command=self.enable_drawing)
        # operation_menu.add_command(label="Disable Drawing", command=self.disable_drawing)
        # operation_menu.add_command(label="Quit", command=self.break_mainloop)

        # Bind the window close button to the quit_me function
        self.root.protocol("WM_DELETE_WINDOW", self.quit_me)

        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="Operation", menu=operation_menu)
        self.root.config(menu=menu_bar)

        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.save_rect(event))
        
        output_button = tk.Button(self.root, text="Output Power", command=self.output_power)
        output_button.pack(side=tk.LEFT)

        # Add a button to show the figure
        show_figure_button = tk.Button(self.root, text="Show Contour", command=self.show_contour)
        show_figure_button.pack(side=tk.LEFT)

        # Add a button to save rectangles data
        show_figure_button = tk.Button(self.root, text="Save Location", command=self.save_location)
        show_figure_button.pack(side=tk.LEFT)

        # Add a button to load rectangles data
        show_figure_button = tk.Button(self.root, text="Load Location", command=self.load_location)
        show_figure_button.pack(side=tk.LEFT)

        self.root.mainloop()

            
        










