import torch
import os
import numpy as np
from classifier.utils import Average
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,f1_score

def eval_annotator_model(annotator_model, x_data, y_data, y_annot_data, inst_annot, device = 'cpu'): ##Not important for Validation Set
    
    x_data = x_data.to_numpy()
    x_data = torch.from_numpy(x_data).to(device)

    y_data = y_data.to_numpy()
    y_data = torch.from_numpy(y_data).to(device)

    y_annot_data = y_annot_data.to_numpy()
    y_annot_data = torch.from_numpy(y_annot_data).to(device)

    x_data = torch.tensor(x_data,dtype=torch.float32).clone().detach()
    y_data = y_data.detach().numpy()

    output = annotator_model(x_data)
    
    MO = []
    WO = []
        
    max_index = torch.argmax(torch.mul(output,inst_annot),dim=1)
    
    m = inst_annot.shape[1]

    for i in range(y_annot_data.shape[0]):
        MO.append(y_annot_data[i][max_index[i]])
        
        average_weights = dict()
        for j in range(m):
            if inst_annot[i][j] == 0:
                continue
            if y_annot_data[i][j].item() in average_weights:
                average_weights[y_annot_data[i][j].item()].append(output[i][j])
            else :
                average_weights[y_annot_data[i][j].item()] = [output[i][j]]
        for key in average_weights:
            average_weights[key] = Average(average_weights[key])
        Keymax = [key for key, value in average_weights.items() if value == max(average_weights.values())][0]
        WO.append(Keymax)
    
    wo = np.array(WO)
    mo = np.array(MO)

    a1 = accuracy_score(wo,y_data)
    f1 = f1_score(wo,y_data)

    a2 = accuracy_score(mo,y_data)
    f2 = f1_score(mo,y_data)


    # print('Metrics for Weighted Label selection ')
    # print(confusion_matrix(wo,y_data))
    # print(classification_report(wo,y_data))
    # print("Accuracy::", accuracy_score(wo,y_data))
    # print("F1 score :: ", f1_score(wo,y_data))

    # print('\n\nMetrics for maximum index selection ')
    # print(confusion_matrix(mo,y_data))
    # print(classification_report(mo,y_data))
    # print("Accuracy::", accuracy_score(mo,y_data))
    # print("F1 score :: ", f1_score(mo,y_data))

    return a1,f1,a2,f2

def annot_eval_after_warmup(annotator_model, BOOT, data_a):
    x_boot, y_boot, y_annot_boot = BOOT
    inst_annot = np.ones_like(y_annot_boot)
    inst_annot = torch.from_numpy(inst_annot)
    a1,f1,a2,f2 = eval_annotator_model(annotator_model, x_boot, y_boot, y_annot_boot,inst_annot)
    data_a[0]['Annotator Accuracy after Warmup on Boot Data'] = a1
    data_a[0]['Annotator F1 Score after Warmup on Boot Data'] = f1
    data_a[1]['Annotator Accuracy after Warmup on Boot Data'] = a2
    data_a[1]['Annotator F1 Score after Warmup on Boot Data'] = f2
    

def annot_eval_after_training(annotator_model,new_active_x,new_active_y_true,new_active_y_annot,new_active_mask,data_a):
    new_active_mask = new_active_mask.to_numpy()
    new_active_mask = torch.from_numpy(new_active_mask)
    a1,f1,a2,f2 = eval_annotator_model(annotator_model, new_active_x, new_active_y_true, new_active_y_annot,new_active_mask)
    data_a[0]['Annotator Accuracy after Training'] = a1
    data_a[0]['Annotator F1 Score after Training'] = f1
    data_a[1]['Annotator Accuracy after Training'] = a2
    data_a[1]['Annotator F1 Score after Training'] = f2

def annotator_AL_loss(loss,Path):

    path = os.path.join(Path,"Annotator_Loss_during_AL.png")
    plt.figure(figsize = (10, 5))
    plt.plot(loss, color ='blue')
    plt.xlabel("AL Cycles") 
    plt.ylabel("Loss")
    plt.title("Annotator Loss during active learning")
    plt.savefig(path)
    plt.show()
   
    

