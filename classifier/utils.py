import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,f1_score

def find_entropy(output):
  base = output.shape[1]
  Entropy = list()
  for i in output:
    Entropy.append(-i[0]*np.log(i[0]) - i[1]*np.log(i[1]))
  return Entropy

def find_index(output,index_frame):
  Entropy = find_entropy(output)
  max_index = np.argmax(Entropy,axis=0)
  #print('max_indexes : ',max_indexes)
  #max_index = random.choice(max_indexes.tolist())
  return index_frame[max_index]

def find_entropy_RL(output,inst_annot,choice):
  base = output.shape[1]
  m = inst_annot.shape[1]
  Entropy = list()
  for iter,i in enumerate(output):
    if choice == 0 :
      if torch.sum(inst_annot[iter]).item() > 0.5: # Explore
        Entropy.append(0)
      else :
        Entropy.append(-i[0]*np.log(i[0]) - i[1]*np.log(i[1]))
    elif choice == 1 :
      if torch.sum(inst_annot[iter]).item() == 0:  # Exploit
        Entropy.append(0)
      else :
        Entropy.append(-i[0]*np.log(i[0]) - i[1]*np.log(i[1]))
        
  return Entropy


def find_index_RL(output,index_frame,inst_annot,choice):
  
  Entropy = find_entropy_RL(output,inst_annot,choice)
  max_index = np.argmax(Entropy,axis=0)
  #max_index = random.choice(max_indexes.tolist())
  return index_frame[max_index]


def Average(lst):
    return sum(lst) / len(lst)
