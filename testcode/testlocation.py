# Import necessary libraries
import os
import tkinter as tk
from PIL import Image,ImageTk

# Define function to print the pixel position when the image is clicked
def print_position(event):
    x, y = event.x, event.y
    print(f"Pixel position: ({x}, {y})")

# Create a tkinter window
window = tk.Tk()

# Create a canvas widget and place it in the window
canvas = tk.Canvas(window, width=320, height=240)
canvas.pack()

# Load the image and place it in the canvas
os.chdir(os.path.dirname(__file__))
image = ImageTk.PhotoImage(file = "Figure/test.jpg")
canvas.create_image(0, 0, anchor=tk.NW, image=image)

# Bind the print_position function to the canvas so that it is called when the canvas is clicked
canvas.bind("<Button-1>", print_position)

# Run the tkinter event loop
window.mainloop()
