# Import Libraries
import matplotlib.pyplot as plt
import numpy as np

# Drawing state
drawing = False

# Creatign 28x28 pixel canvas
canvas = np.zeros((28,28))

# Lists
dataset = [] # stores images
labels = [] # stores correct numbers (ground truth)

# Neural network
input_size = 784
hidden_size = 64
output_size = 10
lr = 0.05

np.random.seed(0)

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)*0.1
b1 = np.zeros((1,hidden_size))
W2 = np.random.randn(hidden_size, output_size)*0.1
b2 = np.zeros((1,output_size))

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

def on_key(event):
    global canvas
    
    # code to clear
    if event.key == 'c':
        canvas[:] = 0
        img.set_data(canvas)
        fig.canvas.draw_idle()
        
    # code to save digit
    elif event.key in "0123456789":

        label = int(event.key)

        # flatten 28x28 into 784
        sample = canvas.flatten()

        dataset.append(sample)
        labels.append(label)

        print("Saved:", label, "| Total samples:", len(dataset))
    
    # key to enter prediction
    elif event.key == 'enter':
        sample = canvas.flatten().reshape(1, -1)

        prediction = predict(sample)[0]

        print("Prediction:", prediction)
    
    # key to train 
    elif event.key == 't':
        if len(dataset) > 0:
            X = np.array(dataset)
            y = np.array(labels)
            train(X, y, epochs=200)
            print("Training complete!")
        else:
            print("No data to train on.")

# Activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e/e.sum(axis=1,keepdims=True)

# Forward pass (gives prediction)
def forward(X):
    a1 = sigmoid(X@W1 + b1)
    a2 = softmax(a1@W2 + b2)
    return a1,a2

# Cost or loss function
def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    
    # One-hot encode labels
    y_onehot = np.zeros((m, 10))
    y_onehot[np.arange(m), y_true] = 1

    # Avoid log(0)
    loss = -np.sum(y_onehot * np.log(y_pred + 1e-8)) / m
    return loss, y_onehot

# Training function
def train(X, y, epochs=200):
    global W1, b1, W2, b2

    for epoch in range(epochs):

        # forward pass
        a1 = sigmoid(X @ W1 + b1)
        a2 = softmax(a1 @ W2 + b2)

        # finding loss
        loss, y_onehot = compute_loss(a2, y)

        m = X.shape[0]

        # backprop

        # Output layer error
        dz2 = (a2 - y_onehot) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer error
        da1 = dz2 @ W2.T
        dz1 = da1 * a1 * (1 - a1)   # sigmoid derivative

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # update weights
        W1 = W1 - (lr * dW1)
        b1 = b1 - (lr * db1)
        W2 = W2 - (lr * dW2)
        b2 = b2 - (lr * db2)

# Prediction function
def predict(X):
    _, probs = forward(X)
    return np.argmax(probs, axis=1)

# canvas plotting
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