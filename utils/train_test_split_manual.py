import numpy as np

def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets manually.

    Parameters:
    - X (np.ndarray): Feature matrix where each row is a sample and each column is a feature.
    - y (np.ndarray): Target vector where each element corresponds to the class label of a sample in X.
    - test_size (float): Proportion of the dataset to include in the test split (between 0 and 1).
    - random_state (int): Seed for the random number generator for reproducibility.

    Returns:
    - X_train (np.ndarray): Training set feature matrix.
    - X_test (np.ndarray): Testing set feature matrix.
    - y_train (np.ndarray): Training set target vector.
    - y_test (np.ndarray): Testing set target vector.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_state)
    
    # Create an array of indices corresponding to the samples in X
    indices = np.arange(X.shape[0])
    
    # Shuffle the indices to randomize the order of samples
    np.random.shuffle(indices)

    # Calculate the number of samples to include in the test set
    test_size = int(X.shape[0] * test_size)

    # Select the first 'test_size' indices for the test set
    test_indices = indices[:test_size]
    
    # Select the remaining indices for the training set
    train_indices = indices[test_size:]

    # Split the feature matrix into training and testing sets using the selected indices
    X_train, X_test = X[train_indices], X[test_indices]
    
    # Split the target vector into training and testing sets using the selected indices
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Example usage:
# X = np.random.rand(100, 5)
# y = np.random.randint(0, 2, 100)
# X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
