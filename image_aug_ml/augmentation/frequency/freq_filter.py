from typing import Tuple

import numpy as np


def gaussian(dims: Tuple[int, int], cutoff_freq: float) -> np.ndarray:
    """Creates gaussian transfer function with dimension <dims> and cutoff frequency <cutoff_freq>

    Parameters
    ----------
    dims : Tuple[int, int]
        dimensions of gaussian transfer function
    cutoff_freq : float
        cutoff frequency for gaussian

    Returns
    -------
    np.ndarray
        gaussian transfer function
    """
    # create grid
    m, n = [(dim - 1) / 2 for dim in dims]
    yy, xx = np.ogrid[-m : m + 1, -n : n + 1]

    # compute transfer function
    tf = np.exp(-(np.power(xx, 2) + np.power(yy, 2)) / (2 * np.power(cutoff_freq, 2)))

    # normalize and return transfer func
    return (tf - np.max(tf)) / (np.max(tf) - np.min(tf))


def freq_filt(orig_img: np.ndarray, transfer_func: np.ndarray) -> np.ndarray:
    """Frequency filters image using transfer function.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to frequency filter
    transfer_func : np.ndarray
        transfer function to apply to original image

    Returns
    -------
    np.ndarray
        frequency filtered image
    """
    # pad and center the input image
    M, N = orig_img.shape[:2]
    padded_img = np.pad(
        orig_img,
        (
            (int(np.floor(M / 2)), int(np.ceil(M / 2))),
            (int(np.floor(N / 2)), int(np.ceil(N / 2))),
            (0, 0),
        ),
        constant_values=0,
    )

    # take fft of image
    f_img = np.fft.fftshift(np.fft.fft2(padded_img.astype(np.float32)))

    # get product of image and transfer func
    f_filtered = np.empty_like(f_img)
    for channel_idx in range(f_img.shape[-1]):
        f_filtered[:, :, channel_idx] = f_img[:, :, channel_idx] * transfer_func

    # get image using ifft
    filtered_img = np.real(np.fft.ifft2(np.fft.fftshift(f_filtered)))

    # slice to remove padding
    filtered_img = filtered_img[
        int(M / 2) : int(3 * M / 2), int(N / 2) : int(3 * N / 2), :
    ]

    # scale and return filtered image
    return (
        255
        * (filtered_img - np.min(filtered_img))
        / (np.max(filtered_img) - np.min(filtered_img))
    ).astype(np.uint8)
