# Import necessary packages.
from src.data_loader import load_data
from src.model import train_random_forest, predict
from src.evaluation import calculate_metrics, get_confusion_matrix
from src.visualize import plot_confusion_matrix

def main():
    # Load data
    x_train, x_test, y_train, y_test = load_data('data/data.csv', target_column='Activity')
    
    # Train model
    model = train_random_forest(x_train, y_train)
    
    # Predict
    y_pred = predict(model, x_test)
    
    # Evaluate
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    
    # Confusion Matrix
    cm = get_confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
