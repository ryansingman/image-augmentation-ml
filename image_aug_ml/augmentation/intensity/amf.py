from typing import Tuple

import numpy as np


def build_kernel(img: np.ndarray, ii: int, jj: int, kernel_size: int) -> np.ndarray:
    """Builds kernel from image at coords ii, jj with provided kernel size.

    Parameters
    ----------
    img : np.ndarray
        input image array
    ii : int
        row coordinate
    jj : int
        col coordinate
    kernel_size : int
        size of kernel

    Returns
    -------
    np.ndarray
        kernel created from image
    """
    # find kernel widths
    kernel_width = int(kernel_size / 2)

    # build and return kernel
    return img[
        max(0, ii - kernel_width) : min(ii + kernel_width, img.shape[0]),
        max(0, jj - kernel_width) : min(jj + kernel_width, img.shape[1]),
    ]


def get_kernel_stats(kernel: np.ndarray) -> Tuple[int, int, int]:
    """Returns minimum, median, and maximum of kernel.

    Parameters
    ----------
    kernel : np.ndaray
        kernel to generate statistics for

    Returns
    -------
    Tuple[int, int, int]
        minimum, median, and maximum value of kernel
    """
    return np.min(kernel), np.median(kernel), np.max(kernel)


def level_b(z_xy: int, z_min: int, z_med: int, z_max: int) -> int:
    """Performs level B of AMF procedure

    Parameters
    ----------
    z_xy : int
        intensity value at center of kernel
    z_min : int
        minimum intensity in kernel
    z_med : int
        median intensity in kernel
    z_max : int
        maximum intensity in kernel

    Returns
    -------
    int
        filtered intensity value
    """
    # if intensity is in range of kernel, return intensity
    if z_min < z_xy < z_max:
        return z_xy

    # otherwise return median
    return z_med


def level_a(
    orig_img: np.ndarray, ii: int, jj: int, kernel_size: int, max_kernel_size: int
) -> int:
    """Performs level A of AMF procedure

    Parameters
    ----------
    orig_img : np.ndarray
        image to perform AMF on
    ii : int
        row coordinate
    jj : int
        col coordinate
    kernel_size : int
        size of median filter kernel
    max_kernel_size : int
        maximum size of median filter kernel

    Returns
    -------
    int
        filtered intensity value at coordinate
    """
    # create kernel
    kernel = build_kernel(orig_img, ii, jj, kernel_size)

    # get kernel statistics
    z_min, z_med, z_max = get_kernel_stats(kernel)

    # if median is between min and max, return level b result
    if z_min < z_med < z_max:
        return level_b(orig_img[ii, jj], z_min, z_med, z_max)

    # increase the kernel size
    kernel_size += 2

    # if kernel size is less than max, repeat level A
    if kernel_size <= max_kernel_size:
        return level_a(orig_img, ii, jj, kernel_size, max_kernel_size)

    # otherwise return median
    return z_med


def amf(
    orig_img: np.ndarray, init_kernel_size: int = 3, max_kernel_size: int = 7
) -> np.ndarray:
    """Performs adaptive median filtering on original image.

    Parameters
    ----------
    orig_img : np.ndarray
        image to perform AMF on
    init_kernel_size : int
        initial size of median filter kernel, by default 3
    max_kernel_size : int
        maximum size of median filter kernel, by default 7

    Returns
    -------
    np.ndarray
        adaptive median filtered image
    """
    # assert that kernel sizes are odd
    assert init_kernel_size % 2 == 1, "Initial kernel size must be odd"
    assert max_kernel_size % 2 == 1, "Max kernel size must be odd"

    # construct output image
    amf_img = np.empty_like(orig_img)

    # iterate over pixels in image
    for ii, jj, kk in np.ndindex(*orig_img.shape):
        # set filtered value of pixel
        amf_img[ii, jj, kk] = level_a(
            orig_img[:, :, kk], ii, jj, init_kernel_size, max_kernel_size
        )

    # return filtered image
    return amf_img
