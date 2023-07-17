from torch.utils.data import Dataset
import torch

class Data(Dataset):
    def __init__(self,x_data,y_data,device= 'cpu'):
        self.x=torch.from_numpy(x_data).to(device)
        self.y=torch.from_numpy(y_data).to(device)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
    

class Data_AL(Dataset):
    def __init__(self,x_data,y_data,mask,device = 'cpu'):
        self.x=torch.from_numpy(x_data).to(device)
        self.y=torch.from_numpy(y_data).to(device)
        self.mask=torch.from_numpy(mask).to(device)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index], self.mask[index]
    def __len__(self):
        return self.len