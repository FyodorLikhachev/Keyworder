from sklearn import metrics
from torch import device
from constants import DEVICE, TRAIN_KEYWORD_MARKER

device = device(DEVICE)

def showMetrics(model ,dataset, title):
    y_pred = []
    y_true = []

    for x, y, c in dataset:
        x = x.to(device)
        y = y.to(device)
        c = c
        pred = get_pred(model, x)

        y_pred.append(pred)
        y_true.append(c)
    
    pos_correct = [y_true == y_pred for y_true, y_pred in zip(y_true, y_pred) if y_true == 1]
    neg_correct = [y_true == y_pred for y_true, y_pred in zip(y_true, y_pred) if y_true == 0]
    
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    
    print(title)
    print("Accuracy: {0} Precision: {1} Recall: {2}".format(round(accuracy, 2), round(precision, 2), round(recall, 2)))
    print("{0} из {1} позитивов".format(sum(pos_correct), len(pos_correct)))
    print("{0} из {1} негативов".format(sum(neg_correct), len(neg_correct)))

def GetAccuracy(model, dataset):
    correct = 0
    for i, (x, y, c) in enumerate(dataset):
        pred = get_pred(model, x)

        if c == pred:
            correct += 1

    return correct/len(dataset)

def get_pred(model, x):
    return 1 if model(x.to(device))[-TRAIN_KEYWORD_MARKER:-1].argmax(dim=1).sum() > 0 else 0
