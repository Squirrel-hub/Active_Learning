from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,f1_score
import matplotlib.pyplot as plt
import os

def eval_model(y_test, y_pred):
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    a = accuracy_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    # print("Accuracy::", a)
    # print("F1 score :: ", f)
    return a,f

def classf_eval_after_warmup(Classifiers,BOOT,VAL,data):
    classifier_model_1,classifier_model_2,classifier_model_3,classifier_model_4 = Classifiers
    x_val, y_val, y_annot_val = VAL
    x_boot, y_boot, y_annot_boot = BOOT
    y_pred_val = classifier_model_1.predict(x_val)
    y_pred_boot = classifier_model_1.predict(x_boot)

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON BOOT')
    a,f = eval_model( y_boot, y_pred_boot)

    data[0]['After Warmup Accuracy on Boot Data'] = a
    data[0]['After Warmup F1 Score on Boot Data'] = f

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON VALIDATION')
    a,f = eval_model( y_val, y_pred_val)

    data[0]['After Warmup Accuracy on Validation Data'] = a
    data[0]['After Warmup F1 Score on Validation Data'] = f

    y_pred_val = classifier_model_2.predict(x_val)
    y_pred_boot = classifier_model_2.predict(x_boot)


    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON BOOT')
    a,f = eval_model( y_boot, y_pred_boot)

    data[1]['After Warmup Accuracy on Boot Data'] = a
    data[1]['After Warmup F1 Score on Boot Data'] = f

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON VALIDATION')
    a,f = eval_model( y_val, y_pred_val)

    data[1]['After Warmup Accuracy on Validation Data'] = a
    data[1]['After Warmup F1 Score on Validation Data'] = f

    y_pred_val = classifier_model_3.predict(x_val)
    y_pred_boot = classifier_model_3.predict(x_boot)

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON BOOT')
    a,f = eval_model( y_boot, y_pred_boot)

    data[2]['After Warmup Accuracy on Boot Data'] = a
    data[2]['After Warmup F1 Score on Boot Data'] = f

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON VALIDATION')
    a,f = eval_model( y_val, y_pred_val)

    data[2]['After Warmup Accuracy on Validation Data'] = a
    data[2]['After Warmup F1 Score on Validation Data'] = f

    y_pred_val = classifier_model_4.predict(x_val)
    y_pred_boot = classifier_model_4.predict(x_boot)

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON BOOT')
    a,f = eval_model( y_boot, y_pred_boot)

    data[3]['After Warmup Accuracy on Boot Data'] = a
    data[3]['After Warmup F1 Score on Boot Data'] = f

    # print('\n\nCLASSIFIER METRICS AFTER WARMING UP ON VALIDATION')
    a,f = eval_model( y_val, y_pred_val)

    data[3]['After Warmup Accuracy on Validation Data'] = a
    data[3]['After Warmup F1 Score on Validation Data'] = f
        
def classf_eval_after_training(Classifiers,new_active_x,new_active_y_opt,new_active_y_majority,new_active_y_true,VAL,data):


    classifier_model_1,classifier_model_2,classifier_model_3,classifier_model_4 = Classifiers
    x_val, y_val, y_annot_val = VAL

    y_pred_val_1 = classifier_model_1.predict(x_val)
    a,f = eval_model(y_pred_val_1,y_val)

    data[0]['After Training Accuracy on Validation Data'] = a
    data[0]['After Training F1 Score on Validation Data'] = f

    y_pred_val_2 = classifier_model_2.predict(x_val)
    a,f = eval_model(y_pred_val_2,y_val)

    data[1]['After Training Accuracy on Validation Data'] = a
    data[1]['After Training F1 Score on Validation Data'] = f

    y_pred_val_3 = classifier_model_3.predict(x_val)
    a,f = eval_model(y_pred_val_3,y_val)

    data[2]['After Training Accuracy on Validation Data'] = a
    data[2]['After Training F1 Score on Validation Data'] = f

    y_pred_val_4 = classifier_model_4.predict(x_val)
    a,f = eval_model(y_pred_val_4,y_val)

    data[3]['After Training Accuracy on Validation Data'] = a
    data[3]['After Training F1 Score on Validation Data'] = f




def compare_true_label(new_active_y,new_active_y_opt,new_active_y_majority,new_active_y_true):
    print('ANNOTATOR MODEL PREDICTED LABELS VS TRUE LABELS')
    print('\nAccuracy : ',accuracy_score(new_active_y,new_active_y_true))
    print('\nF1 Score : ',f1_score(new_active_y,new_active_y_true))
    print('\nConfusion Matrix')
    print(confusion_matrix(new_active_y,new_active_y_true))
    print('\nClassification Report')
    print(classification_report(new_active_y,new_active_y_true))

    print('\n\nW OPTIMAL LABELS VS TRUE LABELS')
    print('\nAccuracy : ',accuracy_score(new_active_y_opt,new_active_y_true))
    print('\nF1 Score : ',f1_score(new_active_y_opt,new_active_y_true))
    print('\nConfusion Matrix')
    print(confusion_matrix(new_active_y_opt,new_active_y_true))
    print('\nClassification Report')
    print(classification_report(new_active_y_opt,new_active_y_true))

    print('\n\nMAJORITY LABELS VS TRUE LABELS')
    print('\nAccuracy : ',accuracy_score(new_active_y_majority,new_active_y_true))
    print('\nF1 Score : ',f1_score(new_active_y_majority,new_active_y_true))
    print('\nConfusion Matrix')
    print(confusion_matrix(new_active_y_majority,new_active_y_true)) 
    print('\nClassification Report')
    print(classification_report(new_active_y_majority,new_active_y_true)) 

def print_scores(Classifiers,VAL):

    classifier_model_1,classifier_model_2,classifier_model_3,classifier_model_4 = Classifiers
    x_val, y_val, y_annot_val = VAL
    
    y_pred_val_1 = classifier_model_1.predict(x_val)
    a,f = eval_model(y_pred_val_1,y_val)


    print('\nLabels from Annotator Model')
    print('Accuracy : ',a)
    print('F1 Score : ',f)
    print(confusion_matrix(y_pred_val_1,y_val))
    print(classification_report(y_pred_val_1,y_val))

    y_pred_val_2 = classifier_model_2.predict(x_val)
    a,f = eval_model(y_pred_val_2,y_val)


    print('\nLabels from W optimal')
    print('Accuracy : ',a)
    print('F1 Score : ',f)
    print(confusion_matrix(y_pred_val_2,y_val))
    print(classification_report(y_pred_val_2,y_val))

    y_pred_val_3 = classifier_model_3.predict(x_val)
    a,f = eval_model(y_pred_val_3,y_val)


    print('\nLabels from Majority')
    print('Accuracy : ',a)
    print('F1 Score : ',f)
    print(confusion_matrix(y_pred_val_3,y_val))
    print(classification_report(y_pred_val_3,y_val))

    y_pred_val_4 = classifier_model_4.predict(x_val)
    a,f = eval_model(y_pred_val_4,y_val)


    print('\nTrue Labels')
    print('Accuracy : ',a)
    print('F1 Score : ',f)
    print(confusion_matrix(y_pred_val_4,y_val))
    print(classification_report(y_pred_val_4,y_val))

def classifier_Val_scores_during_AL(c_a,c_f,Path):
    path = os.path.join(Path,"Classifier_Val_Metrics_during_AL.png")
    plt.figure(figsize = (10, 5))
    plt.plot(c_a, color ='blue',label='classifier accuracy')
    plt.plot(c_f, color ='green',label='classifier f1 score')
    plt.legend()
    plt.xlabel("AL Cycles")
    plt.ylabel("Accuracy")
    plt.title("Classifier Metrics")
    plt.savefig(path)
    plt.show()
    