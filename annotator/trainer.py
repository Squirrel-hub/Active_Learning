import torch
import torch.nn as nn
import numpy as np
import random
import os
import tqdm
import matplotlib.pyplot as plt
from Loaders.dataset_loaders import Data, Data_AL
from torch.utils.data import Dataset, DataLoader
from classifier.utils import Average



def annotator_training(model_annotator, new_active_x, new_active_w, new_active_mask,batch_size = 4,n_epochs=1000, learning_rate = 0.001, device = 'cpu'):
    
    optimizer=torch.optim.Adam(model_annotator.parameters(), lr=learning_rate)
    criterion_annotator = nn.MSELoss()
    data_set=Data_AL(new_active_x.to_numpy(),new_active_w.to_numpy(),new_active_mask.to_numpy())
    trainloader=DataLoader(dataset=data_set,batch_size=batch_size)
    loss_list = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, y, mask in trainloader:
            x = torch.tensor(x,dtype=torch.float32).to(device)
            #clear gradient 
            mask = mask.to(device)
            optimizer.zero_grad()
            #make a prediction 
            z=model_annotator(x)
            y = y.squeeze(1)
            y = y.type(torch.FloatTensor).to(device)
            x1 = torch.mul(mask,z).to(device)
            x2 = torch.mul(mask,y).to(device)
            loss = criterion_annotator(x1,x2)
                
            # calculate gradients of parameters 
            loss.backward()
            epoch_loss += loss.to('cpu').data
            # update parameters 
            optimizer.step()
            
        loss_list.append(epoch_loss/batch_size)
        #print('epoch {}, loss {}'.format(epoch, loss_list[-1]))  
    
    # plt.figure()
    # plt.plot(loss_list)
    # plt.title('Annotator Model Training')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    
    return model_annotator,loss_list



