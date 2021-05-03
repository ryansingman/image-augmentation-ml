import numpy as np

from .freq_filter import freq_filt, gaussian


def lowpass(orig_img: np.ndarray) -> np.ndarray:
    """Lowpass filters image and returns result.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to lowpass filter

    Returns
    -------
    np.ndarray
        lowpass filtered image
    """
    # cutoff frequency generation (from 1 to 25)
    cutoff_freq = np.random.uniform(1, 25)

    # create gaussian transfer function
    M, N = orig_img.shape[:2]
    transfer_func = gaussian((M * 2, N * 2), cutoff_freq)

    # frequency filter and return image
    return freq_filt(orig_img, transfer_func)
