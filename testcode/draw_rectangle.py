import tkinter as tk

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        return abs(self.x2 - self.x1) * abs(self.y2 - self.y1)

    def position(self):
        return (self.x1, self.y1), (self.x2, self.y2)

class App:
    def __init__(self, master):
        self.master = master
        self.rectangles = []
        self.canvas = tk.Canvas(master, width=500, height=500)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.start_x = None
        self.start_y = None
        self.rect = None

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        self.rectangles.append(Rectangle(self.start_x, self.start_y, end_x, end_y))
        self.rect = None

    def calculate(self):
        for r in self.rectangles:
            print("Area:", r.area())
            print("Position:", r.position())

root = tk.Tk()
app = App(root)
button = tk.Button(root, text="Calculate", command=app.calculate)
button.pack()
root.mainloop()

