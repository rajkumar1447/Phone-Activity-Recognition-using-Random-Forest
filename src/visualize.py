# Import Necessary packages.
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot the Confusion matrix.
def plot_confusion_matrix(confusion_m, title='Confusion Matrix'):
    sns.heatmap(confusion_m, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.show()
