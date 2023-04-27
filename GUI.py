import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import subprocess
import threading
from sklearn.linear_model import LinearRegression
import serial
import matplotlib.pyplot as plt
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

        self.mask = None
        self.rect = None
        self.image = None
        self.data = None # temperature data
        self.image_path = None
        self.image_width = None
        self.image_height = None
        # self.key = None
        self.rect_count = 0 # Initialize a counter for the rectangles

    def open_image(self):
        self.__init__()
        self.canvas.delete(tk.ALL)
        self.image_path = filedialog.askopenfilename(filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        image = Image.open(self.image_path)
        self.image_width, self.image_height = image.size
        self.image = ImageTk.PhotoImage(file=self.image_path)
        self.canvas.config(width=self.image_width, height=self.image_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        image_path_without_extension = self.image_path.split('.')[0]
        # Split the image_path_without_extension by '_' and take the first element as the directory path
        directory_path = '_'.join(self.image_path.split('_')[:-1])
        file_path = directory_path + '.txt'
        self.data = np.loadtxt(file_path, delimiter=',')

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
        self.mask = np.zeros((self.image_height, self.image_width))
        self.mask[int(self.rect_start_y):int(self.rect_end_y), int(self.rect_start_x):int(self.rect_end_x)] = 1

        # fill in the rest of mask by inverse distance weights
        # center_x = (self.rect_start_x + self.rect_end_x ) / 2
        # center_y = (self.rect_start_y + self.rect_end_y ) / 2
        # area = ((self.rect_end_y - self.rect_start_y) *  (self.rect_end_x - self.rect_start_x)) 
        list_point =  [(int(self.rect_start_y),int(self.rect_start_x)),(int(self.rect_start_y),int(self.rect_end_x)),(int(self.rect_end_y),int(self.rect_start_x)),(int(self.rect_end_y),int(self.rect_end_x))]
        # d_max = area
        # d_fact = area * d_max
        for i in range(self.image_height):
            for j in range(self.image_width):
                if self.mask[i][j] == 0:
                    #average distance
                    sum_dist=0.0
                    
                    for m,n in list_point:
                        dist = np.sqrt((i - m)**2 + (j - n)**2)
                        sum_dist += dist   
                    # 1/X
                    self.mask[i][j] = 1 / (sum_dist/ 4)
                    
                    # self.mask[i][j] = 1 - distance
                    # if self.mask[i][j] < 0:
                    #     self.mask[i][j] = 0               

        self.rectangles.append((self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y, self.mask))
        self.rect = None
        self.rect_count += 1 # Increment the counter for each new rectangle drawn
        # Add a label with the ordinal number of the rectangle in the middle of the rectangle
        self.canvas.create_text((self.rect_start_x + self.rect_end_x)/2, (self.rect_start_y + self.rect_end_y)/2, text=str(self.rect_count), fill='blue', font=("Purisa", 30))

    def output_power(self):
        # Linear regression to get the energy of each rectangle
        X = []
        for rect in self.rectangles:
            start_x, start_y, end_x, end_y, mask = rect
            X.append(mask.flatten())
        X = np.array(X)
        X = X.transpose()
        y = self.data.flatten()
        # Remove rows where all elements in X are 0
        non_zero_rows = np.any(X, axis=1)
        X = X[non_zero_rows]
        y = y[non_zero_rows]
        
        reg = LinearRegression().fit(X, y)
        rectangle_energy = reg.coef_
        # print(reg.intercept_)
        # Predict the energy consumption using the linear regression model
        y_pred = reg.predict(X)

        # Plot the predicted values against the actual values
        plt.scatter(y, y_pred)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Linear Regression Results')
        # Save the linear regression plot as an image file
        plt.savefig('Figure/linear_regression_plot.png')

        # Calculate the R-squared value
        r_squared = reg.score(X, y)
        print("R-squared value:", r_squared)
        rectangle_energy = rectangle_energy / np.sum(rectangle_energy) * self.total_power

        output_text = ""
        for i in range(len(rectangle_energy)):
            output_text += "Component {}'s power consumption is {}\n".format(i + 1, rectangle_energy[i])
        self.output_window = tk.Toplevel()
        self.output_window.title("Power Consumption Output")
        self.output_text = tk.Text(self.output_window, height=10, width=50)
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

        # Create four buttons for wsad input
        # wsad_frame = tk.Frame(root)
        # wsad_frame.pack(side=tk.LEFT)
        # w_button = tk.Button(wsad_frame, text="W", command=lambda: self.send_key("w"))
        # w_button.pack(side=tk.TOP)
        # s_button = tk.Button(wsad_frame, text="S", command=lambda: self.send_key("s"))
        # s_button.pack(side=tk.BOTTOM)
        # a_button = tk.Button(wsad_frame, text="A", command=lambda: self.send_key("a"))
        # a_button.pack(side=tk.LEFT)
        # d_button = tk.Button(wsad_frame, text="D", command=lambda: self.send_key("d"))
        # d_button.pack(side=tk.RIGHT)

        root.mainloop()
        





            
        










