from sklearn.utils import resample
import numpy as np
from collections import defaultdict

def random_oversample(X, y):
    """
    Performs random oversampling to balance the class distribution in the dataset.
    
    Parameters:
    X (np.ndarray): Feature matrix where each row is a sample and each column is a feature.
    y (np.ndarray): Target vector where each element corresponds to the class label of a sample in X.
    
    Returns:
    np.ndarray, np.ndarray: The resampled feature matrix and target vector.
    """
    
    # Initialize a dictionary to count the number of instances for each class
    class_counts = defaultdict(int)
    
    # Iterate through the target vector and count occurrences of each class
    for label in y:
        class_counts[label] += 1
    
    # Determine the maximum count of any class in the dataset
    max_count = max(class_counts.values())

    # Lists to hold the resampled feature matrix and target vector
    X_resampled, y_resampled = [], []

    # Iterate through each class in the dataset
    for label in class_counts:
        # Select all instances of the current class
        X_class = X[y == label]
        y_class = y[y == label]

        # Resample the instances of the current class to match the maximum count
        # Resampling is done with replacement to ensure the count matches max_count
        X_class_resampled, y_class_resampled = resample(
            X_class, y_class, 
            replace=True,          # Allow resampling with replacement
            n_samples=max_count,   # Number of samples to generate
            random_state=42        # Seed for reproducibility
        )

        # Append the resampled instances to the result lists
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)

    # Vertically stack the resampled feature arrays to form the final resampled feature matrix
    X_resampled = np.vstack(X_resampled)
    
    # Horizontally stack the resampled target arrays to form the final resampled target vector
    y_resampled = np.hstack(y_resampled)

    return X_resampled, y_resampled

# Example usage:
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
# y = np.array([0, 0, 1, 1, 1])
# X_res, y_res = random_oversample(X, y)
# print(X_res)
# print(y_res)
