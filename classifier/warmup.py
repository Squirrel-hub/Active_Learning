import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from annotator.warmup import annotator_warmup
from classifier.evaluation import eval_model
from annotator.evaluation import eval_annotator_model
from annotator.warmup import warmup_weights





def warmedup_classifiers(Classifiers, BOOT,annotator_model,W_optimal,device):
    
    x_boot, y_boot, y_annot_boot = BOOT
    classifier_model_AM,classifier_model_WO,classifier_model_M,classifier_model_TL = Classifiers
    
    input = x_boot.to_numpy()
    input = torch.tensor(input,dtype = torch.float32).to(device)
    output = annotator_model(input)

    warmup_annot_pred_1 = []
    max_index_1 = torch.argmax(output,dim=1)

    warmup_annot_pred_2 = []
    W_optimal = W_optimal.to_numpy()
    W_optimal = torch.from_numpy(W_optimal)
    max_index_2 = torch.argmax(W_optimal,dim=1)

    warmup_annot_pred_3 = []

    warmup_annot_pred_4 = []

    for i in range(x_boot.shape[0]):
        annot_index_1 = max_index_1[i].item()
        warmup_annot_pred_1.append(y_annot_boot.iloc[i,annot_index_1])

        annot_index_2 = max_index_2[i].item()
        warmup_annot_pred_2.append(y_annot_boot.iloc[i,annot_index_2])

        arr = np.array(y_annot_boot.iloc[i])
        unique, counts = np.unique(arr, return_counts=True)
        res = unique[np.argmax(counts)]
        warmup_annot_pred_3.append(res)

        warmup_annot_pred_4.append(y_boot.iloc[i])

    y_boot_annot_pred_1 = pd.Series(warmup_annot_pred_1, index = list(x_boot.index.values))
    classifier_model_AM.train(x_boot,y_boot_annot_pred_1)

    y_boot_annot_pred_2 = pd.Series(warmup_annot_pred_2, index = list(x_boot.index.values))
    classifier_model_WO.train(x_boot,y_boot_annot_pred_2)

    y_boot_annot_pred_3 = pd.Series(warmup_annot_pred_3, index = list(x_boot.index.values))
    classifier_model_M.train(x_boot,y_boot_annot_pred_3)

    y_boot_annot_pred_4 = pd.Series(warmup_annot_pred_4, index = list(x_boot.index.values))
    classifier_model_TL.train(x_boot,y_boot_annot_pred_4)
    
    Classifiers = [classifier_model_AM,classifier_model_WO,classifier_model_M,classifier_model_TL]
    y_boot_annot_pred = [y_boot_annot_pred_1,y_boot_annot_pred_2,y_boot_annot_pred_3,y_boot_annot_pred_4]

    return Classifiers,y_boot_annot_pred
