import torch
import torch.nn as nn

class Annotator_1(nn.Module):
    def __init__(self,D_in,D_out):
        super(Annotator_1,self).__init__()
        self.linear1 = nn.Linear(D_in,D_out)
        
    def forward(self,x):
        x = self.linear1(x)
        x=torch.sigmoid(x)  
        return x
    

class Annotator_2(nn.Module):
    def __init__(self,D_in,H_dim,D_out):
        super(Annotator_2,self).__init__()
        self.linear1 = nn.Linear(D_in,H_dim)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(H_dim,D_out)
        self.linear3 = nn.Linear(H_dim,H_dim)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear3(x)
        # x = self.relu(x)
        x = self.linear2(x)
        x=torch.sigmoid(x)  
        return x
