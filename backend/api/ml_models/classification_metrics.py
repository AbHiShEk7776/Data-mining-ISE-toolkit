import numpy as np
from collections import Counter

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    precisions = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        precision_cls = tp / (tp + fp + 1e-9)
        precisions.append(precision_cls)
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
        fp = sum(np.sum((y_pred == cls) & (y_true != cls)) for cls in classes)
        return tp / (tp + fp + 1e-9)
    else:
        raise ValueError("Unknown average type")

def recall(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    recalls = []
    
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        recall_cls = tp / (tp + fn + 1e-9)
        recalls.append(recall_cls)
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in classes)
        fn = sum(np.sum((y_pred != cls) & (y_true == cls)) for cls in classes)
        return tp / (tp + fn + 1e-9)
    else:
        raise ValueError("Unknown average type")

def f1_score(y_true, y_pred, average='macro'):
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * (p * r) / (p + r + 1e-9)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_index[true], class_to_index[pred]] += 1
    
    return matrix

def get_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, average="macro"),
        "recall": recall(y_true, y_pred, average="macro"),
        "f1_score": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()  # Fixed: removed [0]
    }
