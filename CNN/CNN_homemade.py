#%%

import torch
import torch.nn as nn
from torch.nn import functional as F

generator = torch.Generator().manual_seed(123)

input = torch.randn((1,1,8,8), generator=generator) # N, Cin, L -> (1, 8, 8)

kernel = torch.randn((2,2), generator=generator) # K = 2

def Conv2d_homemade(input, kernel):
    
    # stride = 1 LATER 
    # padding = 0 LATER
    
    sr = len(kernel[0,:])
    sc = len(kernel[:,0])
   
    Batch, C_in, Height, Width = input.shape
    
    new_tensor = torch.zeros((Batch, C_in, Height - sr + 1, Width - sc + 1))
    
    for batch_i in range(input.shape[0]):
        
        current_batch = input[batch_i,:,:,:]
        
        for channel_i in range(input.shape[1]):
            
            current_channel = current_batch[channel_i, :,:]
            n = torch.zeros((Height - sr + 1, Width - sc + 1))
            
            for row in range(current_channel.shape[0] - sr + 1):
                for col in range(current_channel.shape[1] - sc + 1):
                    n[row,col] = current_channel[row:row+sr, col:col+sc].reshape(1,-1) @ kernel.reshape(-1, 1)
            
            new_tensor[batch_i, channel_i, : , :] = n
            
    return new_tensor

z = Conv2d_homemade(input, kernel)

# BatchNorm2d_homemade() LATER

def ReLU(input):
    output = torch.where(input > 0 , input, 0)
    return output

n = ReLU(z)

kernel_max_pool = torch.randn((3,3), generator=generator)

# def MaxPool2d_homemade(input, kernel_max_pool):

sr = len(kernel[0,:])
sc = len(kernel[:,0])

Batch, C_in, Height, Width = input.shape

new_tensor = torch.zeros((Batch, C_in, Height - sr + 1, Width - sc + 1))

for batch_i in range(input.shape[0]):
    
    current_batch = input[batch_i,:,:,:]
    
    for channel_i in range(input.shape[1]):
        
        current_channel = current_batch[channel_i, :,:]
        n_ = torch.zeros((Height - sr + 1, Width - sc + 1))
        
        for row in range(current_channel.shape[0] - sr + 1):
            for col in range(current_channel.shape[1] - sc + 1):
                n_[row,col] = current_channel[row:row+sr, col:col+sc].reshape(1,-1) @ kernel.reshape(-1, 1)
        
        new_tensor[batch_i, channel_i, : , :] = n


    # return new_tensor

# a = MaxPool2d_homemade(n, kernel_max_pool)
                


# %%


m = nn.Conv2d(1, 1, kernel_size=(2,2)) 

x = m(input) 

p = nn.MaxPool2d(kernel_size=(3,3))

x2 = p(x)


# %%
