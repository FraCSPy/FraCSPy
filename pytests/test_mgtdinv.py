import numpy as np
import pytest
from fracspy.location.utils import mgtdinv

def test_mgtdinv_2d():
    # Test case 1: 2D input
    g = np.random.rand(6, 10)  # 6 components, 10 receivers
    result = mgtdinv(g)
    assert result.shape == (6, 6)
    assert np.allclose(np.dot(result, np.dot(g, g.T)), np.eye(6))

def test_mgtdinv_2d_identity():
    # Test case 2: 2D input with identity matrix
    g = np.eye(6)
    result = mgtdinv(g)
    assert np.allclose(result, np.eye(6))

# def test_mgtdinv_2d_known_inverse():
#     # Test case 3: 2D input with known result
#     g = np.array([[1], [-1], [0]])    
#     gtg = np.dot(g, g.T)
#     # this is 
#     # array([[ 1, -1,  0],
#     #        [-1,  1,  0],
#     #        [ 0,  0,  0]])
#     #expected_inv = np.linalg.inv(gtg)
#     expected_inv = array([[ 0.25, -1,  0],
#                           [-1,  1,  0],
#                           [ 0,  0,  0]])
#     result = mgtdinv(g)
#     assert np.allclose(result, expected_inv)

def test_mgtdinv_3d():
    # Test case 4: 3D input
    g = np.random.rand(6, 10, 5)  # 6 components, 10 receivers, 5 grid points
    result = mgtdinv(g)
    assert result.shape == (6, 6, 5)
    for i in range(5):
        assert np.allclose(np.dot(result[:,:,i], np.dot(g[:,:,i], g[:,:,i].T)), np.eye(6))

def test_mgtdinv_invalid_shape():
    # Test case 5: Invalid input shape
    g = np.random.rand(6, 10, 5, 2)  # 4D array, which is invalid
    with pytest.raises(ValueError):
        mgtdinv(g)