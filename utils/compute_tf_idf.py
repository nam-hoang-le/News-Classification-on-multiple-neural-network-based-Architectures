from sklearn.preprocessing import normalize
from collections import defaultdict
import numpy as np

def compute_tf_idf(sentences, max_features=10000):
    """
    Computes the TF-IDF matrix for a list of sentences.

    Parameters:
    - sentences (list of str): List of sentences to compute TF-IDF for.
    - max_features (int): Maximum number of features to include in the TF-IDF matrix.

    Returns:
    - tf_idf_matrix (np.ndarray): TF-IDF matrix with shape (n_samples, max_features).
    """
    # List to store term frequencies for each sentence
    term_freqs = []

    # Dictionary to map each word to a unique index and count document frequencies
    vocab = defaultdict(lambda: len(vocab))
    doc_freq = defaultdict(int)

    # Compute term frequencies and document frequencies
    for sentence in sentences:
        term_freq = defaultdict(int)
        words = sentence.split()
        for word in words:
            term_freq[word] += 1
            doc_freq[word] += 1
        term_freqs.append(term_freq)

    # Select the most common words up to the maximum number of features
    most_common_words = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:max_features]
    vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}

    # Initialize the TF-IDF matrix with zeros
    tf_idf_matrix = np.zeros((len(sentences), max_features))

    # Compute the TF-IDF values for each sentence
    for i, term_freq in enumerate(term_freqs):
        for word, count in term_freq.items():
            if word in vocab:
                tf = count / len(term_freq)  # Term frequency
                idf = np.log(len(sentences) / (1 + doc_freq[word]))  # Inverse document frequency
                tf_idf_matrix[i, vocab[word]] = tf * idf  # Compute TF-IDF value

    # Normalize the TF-IDF matrix to have unit length vectors
    tf_idf_matrix = normalize(tf_idf_matrix, norm='l2')

    return tf_idf_matrix

# Example usage:
# sentences = ["this is a sample", "this is another example example"]
# tf_idf_matrix = compute_tf_idf(sentences, max_features=5)
# print(tf_idf_matrix)
