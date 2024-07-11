import numpy as np

def label_encode(labels):
    """
    Encodes categorical labels as integers.

    Parameters:
    - labels (list or np.ndarray): Array-like object containing categorical labels.

    Returns:
    - encoded_labels (np.ndarray): Array of encoded integer labels.
    - label_mapping (dict): Dictionary mapping original labels to their encoded integer values.
    """
    # Create a mapping from each unique label to an integer index
    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
    
    # Encode each label in the input array using the label mapping
    encoded_labels = [label_mapping[label] for label in labels]

    return np.array(encoded_labels), label_mapping

# Example usage:
# labels = ['cat', 'dog', 'cat', 'fish']
# encoded_labels, label_mapping = label_encode(labels)
# print(encoded_labels)
# print(label_mapping)
