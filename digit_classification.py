# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Drawing state
drawing = False

# Creatign 28x28 pixel canvas
canvas = np.zeros((28,28))

# Function for drawing when holding mouse button
def on_press(event):
    global drawing
    drawing = True

# Function to stop drawing when mouse is released
def on_release(event):
    global drawing
    drawing = False

# Draw when mouse is moving
def on_move(event):
    if drawing and event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)

        # draw brush (3x3)
        for i in range(-1,2):
            for j in range(-1,2):
                if 0 <= y+i < 28 and 0 <= x+j < 28:
                    canvas[y+i][x+j] = 1

        img.set_data(canvas)
        fig.canvas.draw_idle()        

# Function to clear canvas
def on_key(event):
    global canvas
    
    if event.key == 'c':
        canvas[:] = 0
        img.set_data(canvas)
        fig.canvas.draw_idle()

fig, ax = plt.subplots()
img = ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)

ax.set_xlim(0,28)
ax.set_ylim(28,0)

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

