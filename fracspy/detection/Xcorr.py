import numpy as np

def Xcorr(x, y):
    """
    Calculate the negative dot product of two normalized vectors.

    This function normalizes the input vectors and computes 
    the negative sum of the product of their corresponding elements, 
    effectively calculating the negative dot product. 

    Parameters:
    x (np.ndarray): The first input array (vector) to compare.
    y (np.ndarray): The second input array (vector) to compare.

    Returns:
    float: The negative dot product of the normalized vectors.
    
    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> y = np.array([4.0, 5.0, 6.0])
        >>> result = Xcorr(x, y)
        >>> print(result)
        -0.9746318461970762
    """

    # Normalize the input arrays
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    
    # Calculate the negative dot product
    loss = -np.sum(np.multiply(x, y))
    return loss
