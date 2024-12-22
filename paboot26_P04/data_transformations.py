'''data_transformations.py
Philip Booth
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Fall 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    
    '''
    #normalize data by subtracting the mean 
    data_norm = (data - data.min(axis = 0) ) / ( data.max(axis = 0) - data.min(axis=0))
    return data_norm


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    data_center = data - data.mean(axis=0)
    return data_center


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    theta = np.deg2rad(degrees)
    if axis == 'x':
        R3 = np.array([ [1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)] ])
    elif axis == 'y':
        R3 = np.array([ [np.cos(theta),0, np.sin(theta)], [0, 1, 1], [-np.sin(theta), 0, np.cos(theta)] ])
    elif axis == 'z':
        R3 = np.array([ [np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1] ])
    else: 
        raise ValueError("Invalid argument in axis. Please pass x, y, or z into axis parameter")
    return R3

    # data_rotate = (R2 @ axis.T).T
    # return data_rotate
