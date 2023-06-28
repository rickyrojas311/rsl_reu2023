"""
Contains functions for a downsampling function that averages values together 
according to a set factor
"""
import numpy as np

def downsample_average(target_array, factor):
    """
    Downsamples an array based off it's shape and the desired factors
    """
    for i in range(len(factor)):
        arrays = []
        split_array = np.array_split(target_array, factor[i], axis=i)
        for array in split_array: #fix edge behavior
            arrays.append(np.mean(array, axis=i))
        import ipdb; ipdb.set_trace()
        target_array = np.stack(arrays, axis=i)
    return target_array


if __name__ == "__main__":
    A = np.array([[3,2],[4,5],[1,2]])
    print(downsample_average(A, (2, 1)))