# Import the packages and libraries.
from sklearn.ensemble import RandomForestClassifier


# Function to train the model.
def train_random_forest(x_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest Classifier on the training data.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


# Function to make the predictions.
def predict(model, x_test):
    """
    Make predictions on the test set.
    """
    return model.predict(x_test)
