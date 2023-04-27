# Import necessary libraries
import os
import tkinter as tk
from PIL import Image,ImageTk

os.chdir(os.path.dirname(__file__))
# Define function to print "hello world" when button is clicked
def print_hello():
    print("hello world")

# Create a tkinter window
window = tk.Tk()
# Create a test image
photo = ImageTk.PhotoImage(file = "Figure/test.jpg")
photolabel = tk.Label(window, image= photo)
photolabel.pack()
# Create a button widget and place it next to the image
button = tk.Button(window, text="Print Hello", command=print_hello)
button.pack(side=tk.RIGHT)

# Run the tkinter event loop
window.mainloop()
  
# The above code creates a tkinter window with a button widget next to an image. When the button is clicked, it calls the print_hello function which prints "hello world" to the console.