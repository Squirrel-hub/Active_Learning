from annotator.trainer import annotator_training
from Lpp.utils import warmup_weights,weights_optimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def annotator_warmup(model_annotator,x_boot, y_annot_boot,batch_size = 4,n_epochs=1000,learning_rate=0.001, device='cpu'):
    model_annotator = model_annotator.to(device)
    mask = np.ones_like(y_annot_boot)
    W_optimal = weights_optimal(y_annot_boot.to_numpy(),mask)
    W_optimal = pd.DataFrame(W_optimal,index = list(x_boot.index.values))
    mask = pd.DataFrame(mask,index = list(x_boot.index.values))
    model_annotator,loss_list = annotator_training(model_annotator,x_boot,W_optimal,mask,batch_size,n_epochs,learning_rate,device)
    plt.figure()
    plt.plot(loss_list)
    plt.title('Annotator Model Warmup Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    return model_annotator,W_optimal,loss_list

