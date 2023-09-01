#%%

import os 
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Setting up working directory
os.chdir("/Users/tylerwallett/Downloads/cifar-10-batches-py")

# Setting up
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict = unpickle('data_batch_1')
dict2 = unpickle('batches.meta')

targets = torch.tensor(dict[b'labels']).reshape(-1,1)
data = torch.tensor(dict[b'data'], dtype=float) / 255
targets_labels = dict2[b'label_names']

data = data.reshape(10000, 3, 32, 32)
N, C, H, W = data.shape

# Hyperparameters 
# ---------------------------------------------------------

generator = torch.Generator().manual_seed(6303)
batch_size = 512
lr = 1e-03
max_iters = 10000
eval_iter = 500
dropout = 0.5

# --------------------------------------------------------

train_cutoff = int(len(data) * 0.8)
test_val_cutoff = int(len(data) * 0.1)

train = data[:train_cutoff]
val = data[train_cutoff: train_cutoff+test_val_cutoff]
test = data[train_cutoff+test_val_cutoff:]

# Creating batches

def get_batches(data, targets):

    index = torch.randint(0, len(data), (1,batch_size), generator = generator).tolist()

    x_batch = torch.concat([ data[index[0][i]] for i in range(len(index[0])) ]).reshape(len(index[0]),C,32,32)
    y_batch = torch.concat([ targets[index[0][i]] for i in range(len(index[0]))])
    return x_batch, y_batch

x_batch, y_batch = get_batches(data, targets)

# Model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.c1 = nn.Conv2d(3, 16, kernel_size=(2,2), dtype=float)
        self.nl1 = nn.BatchNorm2d(16, dtype=float)
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2))
        self.c2 = nn.Conv2d(16, 32, kernel_size=(2,2), dtype=float)
        self.nl2 = nn.BatchNorm2d(32, dtype=float)
        self.pool2 = nn.AvgPool2d(kernel_size=(2,2))
        self.linear1 = nn.Linear(32 * 7 * 7, 400, dtype=float)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(400, 10, dtype=float)
        
    def forwardpropagation(self, batch_x, batch_y):
        x = self.c1(batch_x)
        x = self.act(x)
        x = self.nl1(x)
        x = self.pool1(x)
        x = self.c2(x)
        x = self.act(x)
        x = self.nl2(x)
        x = self.pool2(x).reshape(batch_size, -1)
        x = self.linear1(x)
        x = self.linear2(self.dropout(x))
        
        probs = F.softmax(x)
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(probs, batch_y)
    
        return probs, loss
    
    def predictions(self, probs):
        out = torch.multinomial(probs, num_samples=1)
        return out
    
model = CNN()
probs, loss = model.forwardpropagation(x_batch, y_batch)

# Backpropagation

optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

for iter in range(max_iters):
    
    if iter % eval_iter == 0:
        print(f"{iter} {loss}")
        
    xs, ys = get_batches(data, targets)
    
    probs, loss = model.forwardpropagation(xs, ys)
    
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    optimizer.step()
    
    if iter == max_iters - 1:
        predictions = model.predictions(probs)
        ys = ys.reshape(-1,1)
        print(confusion_matrix(ys, predictions))
        print(accuracy_score(ys, predictions))
    

os.chdir('/Users/tylerwallett/Documents/Documents/GitHub/Deep_Learning')

from matplotlib.animation import FuncAnimation

# Assuming you have X_test, y_test, and predictions

fig, ax = plt.subplots()
ax.set_title("Truth value: - Prediction: ")

def animate(i):
    ax.clear()
    ax.imshow(xs[i][0])
    ax.imshow(xs[i][1])
    ax.imshow(xs[i][2])
    ax.set_title(f" This is a {targets_labels[ys[i].item()]}")
    ax.axis('off')

ani = FuncAnimation(fig, animate, frames=50, interval=1000)
ani.save('CNN_Classification.gif', writer='ffmpeg')

plt.show()

# %%
