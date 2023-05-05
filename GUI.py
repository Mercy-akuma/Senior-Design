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

# from scipy.optimize import nnls
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
        # # segmentation for PCB
        # self.board_start_x = 48
        # self.board_start_y = 41
        # self.board_end_x = 192
        # self.board_end_y = 145
        self.imagename = None
        self.mask = None
        self.rect = None
        self.image = None
        self.data = None # temperature data
        self.image_path = None
        self.image_width = None
        self.image_height = None
        # self.key = None
        self.rect_count = 0 # Initialize a counter for the rectangles

    def open_image(self, image_path=None):
        self.__init__()
        self.canvas.delete(tk.ALL)
        if image_path:
            self.image_path = image_path.replace("\\", "/")
            
        else:
            self.image_path = filedialog.askopenfilename(filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])

        directory_path = '_'.join(self.image_path.split('_')[:-1])
        self.imagename = directory_path.split('/')[-1]

        file_path = directory_path + '.txt'
        self.data = np.loadtxt(file_path, delimiter=',')

        image = Image.open(self.image_path)
        self.image_width, self.image_height = image.size
        self.image = ImageTk.PhotoImage(file=self.image_path)
        self.canvas.config(width=self.image_width, height=self.image_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # check if the directory exists
        output_directory = os.path.join(os.path.dirname(self.image_path), self.imagename + "_analysis")
        if not os.path.exists(output_directory):
            # if directory does not exist, create it
            os.mkdir(output_directory)

    # def open_text(self):
    #     self.data = None
    #     file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    #     self.data = np.transpose(np.loadtxt(file_path, delimiter=','))
    
    def start_rect(self, event):
        self.rect_start_x = self.canvas.canvasx(event.x)
        self.rect_start_y = self.canvas.canvasy(event.y)

    def draw_rect(self, event):
        self.rect_end_x = self.canvas.canvasx(event.x)
        self.rect_end_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect_start_x = max(0, min(self.rect_start_x, self.image_width))
        self.rect_start_y = max(0, min(self.rect_start_y, self.image_height))
        self.rect_end_x = max(0, min(self.rect_end_x, self.image_width))
        self.rect_end_y = max(0, min(self.rect_end_y, self.image_height))
        self.rect = self.canvas.create_rectangle(self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y, outline='red')

    def save_rect(self, event):
        # Create a mask for the rectangle
        orig_height = self.data.shape[0]
        orig_width = self.data.shape[1]
        trans_height = self.image_height
        trans_width = self.image_width
        # Add a label with the ordinal number of the rectangle in the middle of the rectangle
        self.canvas.create_text((self.rect_start_x + self.rect_end_x)/2, (self.rect_start_y + self.rect_end_y)/2, text=str(self.rect_count + 1), fill='blue', font=("Purisa", 30))
        self.rect_start_y = self.rect_start_y * (orig_height/trans_height)
        self.rect_end_y = self.rect_end_y * (orig_height/trans_height)
        self.rect_start_x = self.rect_start_x * (orig_width/trans_width)
        self.rect_end_x = self.rect_end_y * (orig_width/trans_width)
        self.mask = np.zeros((orig_height, orig_width))
        
        self.mask[int(self.rect_start_y):int(self.rect_end_y), int(self.rect_start_x):int(self.rect_end_x)] = 1

        # fill in the rest of mask by inverse distance weights
        # area = ((self.rect_end_y - self.rect_start_y) *  (self.rect_end_x - self.rect_start_x)) 
        center_y = (self.rect_start_y + self.rect_end_y) / 2
        center_x = (self.rect_start_x + self.rect_end_x) / 2
        radius = np.sqrt((self.rect_end_y - self.rect_start_y)**2 + (self.rect_end_x - self.rect_start_x)**2 ) / 2
        list_point =  [(int(self.rect_start_y),int(self.rect_start_x)),(int(self.rect_start_y),int(self.rect_end_x)),(int(self.rect_end_y),int(self.rect_start_x)),(int(self.rect_end_y),int(self.rect_end_x))]
        for i in range(orig_height):
            for j in range(orig_width):
                if self.mask[i][j] == 0:
                    #average distance
                    sum_dist=0.0
                    
                    for m,n in list_point:
                        dist = np.sqrt((i - m)**2 + (j - n)**2)
                        sum_dist += dist   
                    # 1/X
                    avg_dist = sum_dist/ 4
                    self.mask[i][j] = 1 - math.log(avg_dist / radius)

        self.rectangles.append((self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y, self.mask))
        self.rect = None
        self.rect_count += 1 # Increment the counter for each new rectangle drawn
        
    def output_power(self):
        # Linear regression to get the energy of each rectangle
        X = []
        for rect in self.rectangles:
            start_x, start_y, end_x, end_y, mask = rect
            X.append(mask.flatten())
        X = np.array(X)
        X = X.transpose()
        y = self.data.flatten()
        
        # linear regression
        reg = LinearRegression().fit(X, y)
        rectangle_energy = reg.coef_
        # non-negative least squres
        # rectangle_energy, _ = nnls(X, y)

        # print(reg.intercept_)
        # Predict the energy consumption using the linear regression model
        y_pred = reg.predict(X)

        # Plot the predicted values against the actual values
        plt.figure()
        plt.scatter(y, y_pred)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Linear Regression Results')
        # Save the linear regression plot as an image file
        plt.savefig("Figure/batch_process/" + self.imagename + "_analysis" + "/linear_regression_plot.png")

        # Calculate the R-squared value
        r_squared = reg.score(X, y)
        print("R-squared value:", r_squared)
        rectangle_energy = rectangle_energy / np.sum(rectangle_energy) * self.total_power

        output_text = ""
        for i in range(len(rectangle_energy)):
            output_text += "Component {}'s power consumption is {}\n".format(i + 1, rectangle_energy[i])
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
    
    def enable_drawing(self):
        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.save_rect(event))
    
    def disable_drawing(self):
        self.canvas.bind("<Button-1>", self.show_pixel_position)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def check_mew_image(self):
        # 检测文件夹下最新的jpg文件
        folder_path = 'Figure/batch_process' 
        files = glob.glob(os.path.join(folder_path, '*.jpg')) # 查找所有jpg文件
        if len(files) == 0:
            print("No jpg file found")
            return None
        else:
            latest_file = max(files, key=os.path.getctime) # 找到最新的文件
            print("Latest jpg file found:", latest_file)
        self.open_image(latest_file)

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
        fig.savefig("Figure/batch_process/" + self.imagename + "_analysis"+ "/contour.png")
        # Show the figure in a new window
        img = Image.open("Figure/batch_process/" + self.imagename + "_analysis" + "/contour.png")
        img.show()
    
    # def send_key(self, c): 
    #     key = c

    def run(self):
        root = tk.Tk()
        root.title("Power Consumption GUI")
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Add a label and entry for total_power
        tk.Label(root, text="Total Power:").pack(side=tk.LEFT)
        total_power_entry = tk.Entry(root)
        total_power_entry.pack(side=tk.LEFT)
        # Add a button to set total_power
        set_power_button = tk.Button(root, text="Set Power", command=lambda: setattr(self, 'total_power', float(total_power_entry.get())))
        set_power_button.pack(side=tk.LEFT)

        menu_bar = tk.Menu(root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        operation_menu = tk.Menu(menu_bar, tearoff=0)
        # File_menu
        file_menu.add_command(label="Import Image", command=self.open_image)
        # file_menu.add_command(label="Import Text", command=self.open_text)
        file_menu.add_command(label="Show Image", command=self.show_image)
        file_menu.add_command(label="Check New Image", command=self.check_mew_image)
        # Operation_menu
        # operation_menu.add_command(label="Output Power", command=self.output_power)
        operation_menu.add_command(label="Enable Connection", command=self.cmd_enable)
        operation_menu.add_command(label="Disable Connection", command=self.cmd_disable)
        operation_menu.add_command(label="Enable Drawing", command=self.enable_drawing)
        operation_menu.add_command(label="Disable Drawing", command=self.disable_drawing)

        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="Operation", menu=operation_menu)
        root.config(menu=menu_bar)

        self.canvas.bind("<Button-1>", self.show_pixel_position)
        output_button = tk.Button(root, text="Output Power", command=self.output_power)
        output_button.pack(side=tk.LEFT)

        # Add a button to show the figure
        show_figure_button = tk.Button(root, text="Show Contour", command=self.show_contour)
        show_figure_button.pack(side=tk.LEFT)


        root.mainloop()
        





            
        










