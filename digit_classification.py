# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Drawing function
drawing = False
x_points = []
y_points = []

def on_press(event):
    global drawing, x_points, y_points
    drawing = True
    x_points = []   # start a new line
    y_points = []

def on_release(event):
    global drawing
    drawing = False

def on_move(event):
    if drawing and event.xdata is not None and event.ydata is not None:
        x_points.append(event.xdata)
        y_points.append(event.ydata)

        ax.plot(x_points, y_points, 'b-', linewidth=6.5)
        fig.canvas.draw()

fig, ax = plt.subplots()

ax.set_xlim(0, 28)
ax.set_ylim(0, 28)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()

# Neural network

# Cost or loss function

# Training function

# Backpropagation (Updating weights) function

