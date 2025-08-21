# Import necessary packages.
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load the data.
def load_data(file_path: str, target_column: str = 'Activity', test_size: float = 0.2, random_state: int = 42):
    """
    Load CSV data, split into features and target, and train-test split.
    """
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
