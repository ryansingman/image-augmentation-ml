import numpy as np

from .freq_filter import freq_filt, gaussian


def bandpass(orig_img: np.ndarray) -> np.ndarray:
    """Bandpass filters image and returns result.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to bandpass filter

    Returns
    -------
    np.ndarray
        bandpass filtered image
    """
    # cutoff frequencies generation (from 1 to 25)
    low_cutoff_freq = np.random.uniform(1, 12.5)
    high_cutoff_freq = np.random.uniform(low_cutoff_freq + 5, 25)

    # create gaussian transfer function
    M, N = orig_img.shape[:2]
    transfer_func = gaussian((M * 2, N * 2), high_cutoff_freq) - gaussian(
        (M * 2, N * 2), low_cutoff_freq
    )

    # frequency filter and return image
    return freq_filt(orig_img, transfer_func)
