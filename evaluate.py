import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

sns.set(style="whitegrid")

def evaluate_models(y_test, predictions, ann_info, output_file="outputs/results.csv"):
    results = []

    # Create folders if not exist
    if not os.path.exists("outputs/figures"):
        os.makedirs("outputs/figures")

    # Evaluate ML models
    for name, y_pred in predictions.items():
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        results.append([name, acc, f1, recall, precision, roc_auc, mcc])

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'outputs/figures/{name}_confusion_matrix.png')
        plt.close()

    # Evaluate ANN
    y_pred_ann, history, _ = ann_info
    acc = accuracy_score(y_test, y_pred_ann)
    f1 = f1_score(y_test, y_pred_ann)
    recall = recall_score(y_test, y_pred_ann)
    precision = precision_score(y_test, y_pred_ann)
    roc_auc = roc_auc_score(y_test, y_pred_ann)
    mcc = matthews_corrcoef(y_test, y_pred_ann)
    results.append(['ANN', acc, f1, recall, precision, roc_auc, mcc])

    # ANN confusion matrix
    cm = confusion_matrix(y_test, y_pred_ann)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('ANN Confusion Matrix')
    plt.savefig('outputs/figures/ANN_confusion_matrix.png')
    plt.close()

    # Save results
    results_df = pd.DataFrame(results, columns=['Model','Accuracy','F1','Recall','Precision','ROC-AUC','MCC'])
    results_df.to_csv(output_file, index=False)

    # Display table using tabulate
    print("\n================== Model Evaluation Results ==================")
    print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("==============================================================\n")

    # ANN training curves
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('ANN Accuracy Curve')
    plt.savefig('outputs/figures/ANN_accuracy_curve.png')
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.title('ANN Loss Curve')
    plt.savefig('outputs/figures/ANN_loss_curve.png')
    plt.close()

    return results_df
