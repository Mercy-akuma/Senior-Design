import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import subprocess

class GUI:
    def __init__(self):
        self.rectangles = []
        self.total_power = None # Initialize total_power to None
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None
        self.rect = None
        self.image = None
        self.data = None
        self.image_path = None
        self.image_width = None
        self.image_height = None
        self.rect_count = 0 # Initialize a counter for the rectangles

    def open_image(self):
        self.__init__()
        self.canvas.delete(tk.ALL)
        self.image_path = filedialog.askopenfilename(filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),])
        image = Image.open(self.image_path)
        self.image_width, self.image_height = image.size
        self.image = ImageTk.PhotoImage(file=self.image_path)
        self.canvas.config(width=self.image_width, height=self.image_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def open_text(self):
        self.data = None
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        self.data = np.transpose(np.loadtxt(file_path, delimiter=','))

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
        self.rectangles.append((self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y))
        self.rect = None
        self.rect_count += 1 # Increment the counter for each new rectangle drawn
        # Add a label with the ordinal number of the rectangle in the middle of the rectangle
        self.canvas.create_text((self.rect_start_x + self.rect_end_x)/2, (self.rect_start_y + self.rect_end_y)/2, text=str(self.rect_count), fill='blue', font=("Purisa", 30))

    def output_power(self):
        power_list = []
        for rect in self.rectangles:
            start_x, start_y, end_x, end_y = rect
            if start_x > end_x:
                temp = start_x
                start_x = end_x
                end_x = temp
            if start_y > end_y:
                temp = start_y
                start_y = end_y
                end_y = temp
            sum = 0
            for i in range(int(start_x), int(end_x)):
                for j in range(int(start_y), int(end_y)):
                    sum += self.data[i][j]
            power_list.append(sum / (end_x - start_x) / (end_y - start_y))
        sum_temperature = np.sum(power_list)
        power_list = [power / sum_temperature * self.total_power for power in power_list]
        output_text = ""
        for i in range(len(power_list)):
            output_text += "Component {}'s power consumption is {}\n".format(i + 1, power_list[i])
        self.output_window = tk.Toplevel()
        self.output_window.title("Power Consumption Output")
        self.output_text = tk.Text(self.output_window, height=10, width=50)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.output_text.insert(tk.END, output_text)
        self.scrollbar = tk.Scrollbar(self.output_window)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar.config(command=self.output_text.yview)
        self.output_text.config(yscrollcommand=self.scrollbar.set)
        
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
        file_menu.add_command(label="Import Text", command=self.open_text)
        # Operation_menu
        operation_menu.add_command(label="Output Power", command=self.output_power)
        operation_menu.add_command(label="Enable Connection", command=self.cmd_enable)
        operation_menu.add_command(label="Disable Connection", command=self.cmd_disable)

        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_cascade(label="Operation", menu=operation_menu)
        root.config(menu=menu_bar)

        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.save_rect(event))
            
        root.mainloop()










