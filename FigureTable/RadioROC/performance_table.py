import csv
import numpy as np

def get_diag(idx): # radiologists
    result = []
    with open('kappa/n{}.csv'.format(idx)) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            result.append(int(row['Diagnosis Label']))
    return result

def get_diag(csvfile): # MRI model
    result = []
    with open(csvfile, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['ADD'] != '':
                result.append(int(row['ADD_pred']))
    return result

labels = [1 for _ in range(25)] + [0 for _ in range(25)]

def accuracy(label, diag):
    correct = 0
    for l, d in zip(label, diag):
        if l == d:
            correct += 1
    return float(correct) / 50.0

def sensitivity(label, diag):
    tp, fp, tn, fn = 0, 0, 0, 0
    for l, d in zip(label, diag):
        if l == 0 and d == 0:
            tn += 1
        if l == 1 and d == 1:
            tp += 1
        if l == 0 and d == 1:
            fp += 1
        if l == 1 and d == 0:
            fn += 1
    return float(tp) / (tp + fn + 0.0000001)

def specificity(label, diag):
    tp, fp, tn, fn = 0, 0, 0, 0
    for l, d in zip(label, diag):
        if l == 0 and d == 0:
            tn += 1
        if l == 1 and d == 1:
            tp += 1
        if l == 0 and d == 1:
            fp += 1
        if l == 1 and d == 0:
            fn += 1
    return float(tn) / (tn + fp + 0.0000001)

def f_1(label, diag):
    tp, fp, tn, fn = 0, 0, 0, 0
    for l, d in zip(label, diag):
        if l == 0 and d == 0:
            tn += 1
        if l == 1 and d == 1:
            tp += 1
        if l == 0 and d == 1:
            fp += 1
        if l == 1 and d == 0:
            fn += 1
    return 2 * float(tp) / (2 * tp + fp + fn + 0.0000001)

def mcc(label, diag):
    tp, fp, tn, fn = 0, 0, 0, 0
    for l, d in zip(label, diag):
        if l == 0 and d == 0:
            tn += 1
        if l == 1 and d == 1:
            tp += 1
        if l == 0 and d == 1:
            fp += 1
        if l == 1 and d == 0:
            fn += 1
    return (tp*tn-fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5 + 0.00000001)

def mean_std_accuracy(mode='model'):
    ACC = []
    if mode != 'model':
        for i in range(1, 8):
            diags = get_diag(i)
            acc = accuracy(labels, diags)
            ACC.append(acc)
    else:
        for i in range(5):
            diags = get_diag('../../tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i))
            acc = accuracy(labels, diags)
            ACC.append(acc)
    ACC = np.array(ACC)
    print("accuracy mean:", np.mean(ACC), " std:", np.std(ACC))

def mean_std_sensitivity(mode='model'):
    ACC = []
    if mode != 'model':
        for i in range(1, 8):
            diags = get_diag(i)
            acc = sensitivity(labels, diags)
            ACC.append(acc)
    else:
        for i in range(5):
            diags = get_diag('../../tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i))
            acc = sensitivity(labels, diags)
            ACC.append(acc)
    ACC = np.array(ACC)
    print("sensitivity:", np.mean(ACC), " std:", np.std(ACC))

def mean_std_specificity(mode='model'):
    ACC = []
    if mode != 'model':
        for i in range(1, 8):
            diags = get_diag(i)
            acc = specificity(labels, diags)
            ACC.append(acc)
    else:
        for i in range(5):
            diags = get_diag('../../tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i))
            acc = specificity(labels, diags)
            ACC.append(acc)
    ACC = np.array(ACC)
    print("specificity:", np.mean(ACC), " std:", np.std(ACC))

def mean_std_f1(mode='model'):
    ACC = []
    if mode != 'model':
        for i in range(1, 8):
            diags = get_diag(i)
            acc = f_1(labels, diags)
            ACC.append(acc)
    else:
        for i in range(5):
            diags = get_diag('../../tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i))
            acc = f_1(labels, diags)
            ACC.append(acc)
    ACC = np.array(ACC)
    print("f1:", np.mean(ACC), " std:", np.std(ACC))

def mean_std_mcc(mode='model'):
    ACC = []
    if mode == 'doctor':
        for i in range(1, 8):
            diags = get_diag(i)
            acc = mcc(labels, diags)
            ACC.append(acc)
    elif mode == 'MRI':
        for i in range(5):
            diags = get_diag('../../tb_log/CNN_3ways_special1_cross{}/neuro_test_eval_after.csv'.format(i))
            acc = mcc(labels, diags)
            ACC.append(acc)
    elif mode == 'nonImg':
        pass
    ACC = np.array(ACC)
    print("mcc:", np.mean(ACC), " std:", np.std(ACC))

mean_std_accuracy()
mean_std_sensitivity()
mean_std_specificity()
mean_std_f1()
mean_std_mcc()


