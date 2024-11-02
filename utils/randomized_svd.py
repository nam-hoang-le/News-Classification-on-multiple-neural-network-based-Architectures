import numpy as np

def randomized_svd(X, n_components=2, n_oversamples=10, n_iter=5, random_state=None):
    """
    Performs Singular Value Decomposition (SVD) using a randomized algorithm and returns the reduced-dimensional data.

    Parameters:
    - X (np.ndarray): Data matrix with shape (n_samples, n_features).
    - n_components (int): Number of principal components to retain.
    - n_oversamples (int): Additional number of random samples to ensure quality of the approximation (typically n_components + n_oversamples).
    - n_iter (int): Number of iterations for the power method to improve the approximation accuracy.
    - random_state (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - X_reduced (np.ndarray): Reduced-dimensional data matrix with shape (n_samples, n_components).
    """
    # Set the random seed for reproducibility if random_state is provided
    if random_state is not None:
        np.random.seed(random_state)

    # Determine the total number of random vectors to use
    n_random = n_components + n_oversamples

    # Generate a random Gaussian matrix Q with shape (n_features, n_random)
    Q = np.random.normal(size=(X.shape[1], n_random))

    # Compute the sample matrix Y by multiplying X with Q
    Y = np.dot(X, Q)

    # Perform power iterations to improve the approximation
    for _ in range(n_iter):
        Y = np.dot(X, np.dot(X.T, Y))

    # Compute the orthonormal matrix Q using QR decomposition
    Q, _ = np.linalg.qr(Y)

    # Project the original data matrix X onto the lower-dimensional subspace
    B = np.dot(Q.T, X)

    # Perform SVD on the smaller matrix B
    Uhat, s, Vt = np.linalg.svd(B, full_matrices=False)

    # Compute the final left singular vectors by multiplying Q with Uhat
    U = np.dot(Q, Uhat)

    # Compute the reduced-dimensional data matrix
    X_reduced = np.dot(U[:, :n_components], np.diag(s[:n_components]))

    return X_reduced

# Example usage:
# X = np.random.rand(100, 50)
# X_reduced = randomized_svd(X, n_components=5, random_state=42)
# print(X_reduced)
