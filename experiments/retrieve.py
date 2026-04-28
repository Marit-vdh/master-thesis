import numpy as np


def cosine_similarity(a, b):
    """Compute the cosine similarity between two arrays.

    Args:
        a (np.array): _description_
        b (np.array): _description_

    Returns:
        dot product (float): _description_
    """
    # Step 1: Compute the dot product
    dot_product = np.dot(a, b)

    # Step 2: Compute the magnitudes of the vectors
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Step 3: Calculate cosine similarity
    return dot_product / (magnitude_a * magnitude_b)
