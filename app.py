# Import necessary packages.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/data.csv")

# Preprocess the dataset.
x = df.drop('Activity', axis=1)
y = df['Activity']

# Split the dataset into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make Prediction on the test set.
y_pred = model.predict(x_test)

# Evaluate the model using accuracy, precision, recall, f1-score.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_Score: {f1 * 100:.2f}%")


# Visualize the confusion matrix using Seaborns heatmap.
confusion_m = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_m, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()