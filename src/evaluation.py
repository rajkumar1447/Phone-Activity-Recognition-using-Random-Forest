# Import necessary and required packages.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to calculate the metrics.
def calculate_metrics(y_true, y_pred):
    """
    Calculate standard evaluation metrics for classification.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Function to build the confusion_matrix.
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
