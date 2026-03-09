import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Computes accuracy, precision, and recall.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    }

def save_confusion_matrix(y_true, y_pred, filepath, title="Confusion Matrix"):
    """
    Generates and saves a confusion matrix heatmap as a PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()

def save_metrics(metrics_dict, filepath):
    """
    Saves the computed metrics dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
