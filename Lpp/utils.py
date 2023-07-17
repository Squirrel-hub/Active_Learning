import numpy as np
from scipy.optimize import linprog
import torch

def coeff_annot(arr,idx,inst_annot):
    ones = 0
    m = len(inst_annot)
    
    coefficients = [1 for i in range(m)]

    for i in range(len(arr)):
        if inst_annot[idx] == 0 :
            # print('inst_annot[i] : ',inst_annot[i])
            coefficients[idx] = 0
            coefficients[i] = 0
        elif inst_annot[i] == 0:
            coefficients[i] = 0
        else :
            if i == idx:
                coefficients[idx] = sum(inst_annot)-1
                continue
            if arr[i] == arr[idx] :
                coefficients[i] = -1
            else : 
                coefficients[i] = 1
                ones = ones + 1
    # print('coefficients : ',coefficients)
    # print('ones : ',ones)
    return coefficients,ones

def warmup_weights(y_annot_boot):
    W_optimal = []
    A_ub_list = []
    b_ub_list = []
    masks = []
    #print(type(y_annot_boot))

    m = y_annot_boot.shape[1]
    for i in range(m):
        arr1 = [0 for i in range(m)]
        arr1[i] = 1
        A_ub_list.append(arr1)
        arr2 = [0 for i in range(m)]
        arr2[i] = -1
        A_ub_list.append(arr2)
        b_ub_list.append(1)
        b_ub_list.append(0)


    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    c    = np.array([-1 for i in range(m)])

    inst_annot = [1 for i in range(m)]

    for i in y_annot_boot.to_numpy():
        A_eq_list = []
        b_eq_list = []
        for idx in range(m):
            coefficients,ones = coeff_annot(i,idx,inst_annot)
            A_eq_list.append(coefficients)
            b_eq_list.append(ones)
        
        A_eq = np.array(A_eq_list)
        b_eq = np.array(b_eq_list)

        # Solve linear programming problem
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        W_optimal.append(res.x)
        masks.append([1 for k in range(m)])
        
    return W_optimal,masks

def weights_optimal(y_annot,inst_annot):
    W_optimal = []
    A_ub_list = []
    b_ub_list = []
    masks = []
    #print(type(y_annot_boot))

    m = inst_annot.shape[1]
    for i in range(m):
        arr1 = [0 for i in range(m)]
        arr1[i] = 1
        A_ub_list.append(arr1)
        arr2 = [0 for i in range(m)]
        arr2[i] = -1
        A_ub_list.append(arr2)
        b_ub_list.append(1)
        b_ub_list.append(0)


    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    c    = np.array([-1 for i in range(m)])


    for i,j in zip(y_annot,inst_annot):
        A_eq_list = []
        b_eq_list = []
        for idx in range(m):
            coefficients,ones = coeff_annot(i,idx,j)
            A_eq_list.append(coefficients)
            b_eq_list.append(ones)
        
        A_eq = np.array(A_eq_list)
        b_eq = np.array(b_eq_list)

        # Solve linear programming problem
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        W_optimal.append(res.x)
        masks.append(j)
        
    return W_optimal

def optimal_weights(arr,inst_annot):
    A_ub_list = []
    b_ub_list = []
    m = len(inst_annot)
    

    for i in range(m):
        arr1 = [0 for i in range(m)]
        arr1[i] = 1
        A_ub_list.append(arr1)
        arr2 = [0 for i in range(m)]
        arr2[i] = -1
        A_ub_list.append(arr2)
        b_ub_list.append(1)
        b_ub_list.append(0)

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    c    = np.array([-1 for i in range(m)])

    A_eq_list = []
    b_eq_list = []
    for idx in range(m):
        coefficients,ones = coeff_annot(arr,idx,inst_annot)
        A_eq_list.append(coefficients)
        b_eq_list.append(ones)
    
    A_eq = np.array(A_eq_list)
    b_eq = np.array(b_eq_list)
    # Solve linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    return res.x