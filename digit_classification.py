# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Drawing function
drawing = False
x_points = []
y_points = []

# Function for drawing when holding mouse button
def on_press(event):
    global drawing, x_points, y_points
    drawing = True
    x_points = []   # start a new line
    y_points = []

# Function to stop drawing when mouse is released
def on_release(event):
    global drawing
    drawing = False

# Draw when mouse is moving
def on_move(event):
    if drawing and event.xdata is not None and event.ydata is not None:
        x_points.append(event.xdata)
        y_points.append(event.ydata)

        ax.plot(x_points, y_points,"black", linewidth=6.5)
        fig.canvas.draw()

# Function to clear canvas
def on_key(event):
    global x_points, y_points
    
    if event.key == 'c':   # press 'c' to clear
        ax.cla()           # clear the axes
        ax.set_xlim(0, 28)
        ax.set_ylim(0, 28)
        
        x_points = []
        y_points = []
        
        fig.canvas.draw()



fig, ax = plt.subplots()

ax.set_xlim(0, 28)
ax.set_ylim(0, 28)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.title('Press "C" to clear')

plt.show()

# Neural network

# Cost or loss function

# Training function

# Backpropagation (Updating weights) function

