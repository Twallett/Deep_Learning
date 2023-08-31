#%%

import os 
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt

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
data = torch.tensor(dict[b'data'])
targets_labels = dict2[b'label_names']

#%%

# Visualizing data
for i in range(5):
    plt.imshow(data[i][:1024].reshape(32,32), cmap='Reds', alpha=0.3)
    plt.imshow(data[i][1024:2048].reshape(32,32), cmap='Blues', alpha=0.3)
    plt.imshow(data[i][2048:].reshape(32,32), cmap='Greens', alpha=0.3)
    plt.title(f" This is a {targets_labels[targets[i].item()]}")
    plt.axis(False)
    plt.show()
    
#%%


