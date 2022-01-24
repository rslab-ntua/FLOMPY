import os
import numpy as np
import rasterio
from scipy import signal


def equation_3(fpaths, kernel_N_weight):
    """ Square root of gradient between 2 pixels, in selected orientantion,
    for the whole image. Equation (3).
    After the convolution, the result image will have a pad which must be ignored to fearther
    computations.
    --> For all bands in an fpaths, for one kernel and corresponding weight. All kernels are flipped
    before convolution.

    Args:
    fpaths (string): List of paths for bands b03, b04, b08, b11, b12.
    kernel_N_weight (dataframe): Row of iterable dataframe with 2 columns with kernels and their weights. 

    Return:
    dweek_onePixel (list of 2d array): Result for one week, for one pixel (orientation of convolution).
    """
    #print("equation_3()...Computing bands...")
    res = []
    for im_path in fpaths:
        with rasterio.open(im_path) as src:
            b = src.read(1)
        
            grad = signal.convolve2d(
                b, np.flip(kernel_N_weight['kernels'], 0), mode='same', boundary='fill', fillvalue=0)

            pow_grad = np.power(grad, 2)

            res.append(pow_grad)

    dweek_onePixel = np.sqrt(sum(res))
    return dweek_onePixel


def equation_4(ndvi, kernel_N_weight):
    """
    Args:
    ndvi (2d array)
    Returns:
    weight_abs_grad (list of 2d arrays): as many arrays as kernels
    """
    #print("equation_4()...Computing ndvis...")

    abs_grad = np.absolute(
        signal.convolve2d(
            ndvi, np.flip(kernel_N_weight['kernels'], 0), mode='same', boundary='fill', fillvalue=0))

    return abs_grad


def equation_5(ndvi, dweeks, dndvi, kernels_N_weights):
    """
    Args: 
    dweeks (list of arrays): result of equation_3
    dndvi (list of arrays): result of equation_4
    Returns:
    est (ndarray): Estimation of agricultural edge.
    """
    #print("equation_5()...Computing estimation...")
    # Compute every convolutioned image with it's corresponding weight.
    bands = []
    indices = []
    for i in range(0, len(kernels_N_weights)):
        temp01 = dweeks[i] * kernels_N_weights['weights'][i]
        temp02 = dndvi[i] * kernels_N_weights['weights'][i]
        bands.append(temp01)
        indices.append(temp02)

    # Sum of weights.
    sw = sum(kernels_N_weights['weights'])
    # Return their multiply, with ndvi as the most important contributor.
    return ndvi**2 * sum(bands)/sw * sum(indices)/sw


def edge_cube(listOfPaths):
    # Read metadata of random image
    with rasterio.open(listOfPaths[0], 'r') as src:
        metadata = src.meta

    # Preallocate a zero array with corresponding dimensions
    temp = np.zeros((1, metadata['height'], metadata['width']))

    # Stack arrays as cube
    band_names = []
    for bandpath in listOfPaths:
        with rasterio.open(bandpath, 'r') as src:
            arr = src.read()
            descr = src.name
            band_names.append(os.path.basename(descr))
        
        cbarr = np.concatenate([temp, arr])
        temp = cbarr

    # Update metadata. Reduce one because of temp array
    metadata.update(count=cbarr.shape[0]-1)
    return cbarr[1:, :, :], metadata, band_names