import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import warnings
import os

def diverse_samples(inst_annot,full,Path):
    path = os.path.join(Path,"Diverse_Samples.png")
    unique, frequency = np.unique(inst_annot.sum(1),return_counts=True)
    unique = unique.tolist()
    frequency = frequency.tolist()
    num = [int(i) for i in unique]
    num.append(inst_annot.shape[1])
    frequency.append(full)
    print(num,frequency)

    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(num, frequency, color ='yellow',
            width = 0.4)
    
    plt.xlabel("No. of Annotators Queried")
    plt.ylabel("No. of Instances")
    plt.title("Diversity of samples")
    plt.savefig(path)
    plt.show()


def Knowledge_Base_Metrics(Knowledge_Base,similar_instances,new_active_y,Parent_dir):

    data_KB = [{} for i in range(len(Knowledge_Base.keys()))]
    data_Sim = [{} for i in range(len(similar_instances.keys()))]
    # df_c_KB = pd.DataFrame(data_c_KB,index = ['Annotator Model', 'W Optimal', 'Majority','True Labels'])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        print('\nKNOWLEDGE BASE METRIC')
        i = 0
        for key in Knowledge_Base:
            i = i + 1
            ground_truth = []
            for idx in Knowledge_Base[key]['index']:
                ground_truth.append(new_active_y.loc[idx])
            print('\nSize of Annotator Knowledge Base ',i,' : ',len(ground_truth))
            print('Accuracy Score of Annotator Knowledge Base ',i," : ",accuracy_score(Knowledge_Base[key]['label'],ground_truth))
            print('F1 Score of Annotator Knowledge Base ',i," : ",f1_score(Knowledge_Base[key]['label'],ground_truth))
            data_KB[i-1]['No. of Instances'] = len(ground_truth)
            data_KB[i-1]['Accuracy'] = accuracy_score(Knowledge_Base[key]['label'],ground_truth)
            data_KB[i-1]['f1-score'] = f1_score(Knowledge_Base[key]['label'],ground_truth)
        
        df_KB = pd.DataFrame(data_KB, index = Knowledge_Base.keys())
            

        print('\n\nSIMILAR INSTANCES METRIC')
        i = 0
        for inst in similar_instances:
            i = i + 1
            ground_truth = []
            for idx in similar_instances[inst]['index']:
                ground_truth.append(new_active_y.loc[idx])
            labels = [similar_instances[inst]['label'] for j in range(len(ground_truth))]
            print('\nNumber of similar instances to instance ',inst," : ",len(ground_truth))
            print('Accuracy Score of instances similar to instance ',inst," : ",accuracy_score(labels,ground_truth))
            print('F1 Score of instances similar to instance ',inst," : ",f1_score(labels,ground_truth))
            data_Sim[i-1]['No. of similar Instances'] = len(ground_truth)
            data_Sim[i-1]['Accuracy'] = accuracy_score(labels,ground_truth)
            data_Sim[i-1]['f1-score'] = f1_score(labels,ground_truth)
        df_Sim = pd.DataFrame(data_Sim, index = similar_instances.keys())

        dir1 = "Knowledge_Base_Metrics.csv"
        dir2 = "Similar_Instances_Metrics.csv"

        path1 = os.path.join(Parent_dir,dir1)
        path2 = os.path.join(Parent_dir,dir2)

        df_KB.to_csv(path1)
        df_Sim.to_csv(path2)