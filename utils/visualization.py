import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_feature_distributions(data, predictions=None):
    # Convert the data to a DataFrame for easier handling if it's not already
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))

    # Plot feature distributions from actual data (X_test)
    for i, column in enumerate(data.columns, 1):
        plt.subplot(len(data.columns)//2 + 1, 2, i)
        sns.histplot(data[column], kde=True, color='blue', label='Actual Data')

        if predictions is not None:
            # Only plot predictions on the dropout column
            sns.histplot(predictions['dropout'], kde=True, color='red', label='Predicted Data')

        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()
