import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, name):

    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    path = f"artifacts/{name}_confusion_matrix.png"
    plt.savefig(path)
    plt.close()

    return path

def plot_model_comparison(metrics):

    names = list(metrics.keys())
    scores = list(metrics.values())

    plt.figure()
    plt.bar(names, scores, color=['blue', 'orange'])
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')

    path = "artifacts/model_comparison.png"
    plt.savefig(path)
    plt.close()

    return path
