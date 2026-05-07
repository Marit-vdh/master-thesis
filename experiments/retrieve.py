import numpy as np
from math import dist

def cosine_similarity(a, b):
    """Compute the cosine similarity between two arrays.

    Args:
        a (np.array): array representing the embedding of a paragraph of text.
        b (np.array): array representing the embedding of a prompt.

    Returns:
        cosine similarity (float): represents the degree of which two vectors
        point in the same direction, creating a measure of similarity on an 
        interval of -1 and 1.
    """
    # Step 1: Compute the dot product
    dot_product = np.dot(a, b)

    # Step 2: Compute the magnitudes of the vectors
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Step 3: Calculate cosine similarity
    return dot_product / (magnitude_a * magnitude_b)

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two arrays, which is the square
    root of the sum of squares of the elements of the arrays.

    Args:
        a (np.array): array representing the embedding of a paragraph of text.
        b (np.array): array representing the embedding of a prompt.

    Returns:
        Euclidean distance (float): Represents the distance between two points.
    """
    return dist(a,b)