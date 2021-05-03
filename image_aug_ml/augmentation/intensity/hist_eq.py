import numpy as np


def hist_eq(orig_img: np.ndarray) -> np.ndarray:
    """Performs histogram equalization on the provided image

    Parameters
    ----------
    orig_img : np.ndarray
        image to histogram equalize

    Returns
    -------
    np.ndarray
        histogram equalized image
    """
    # create normalized cumulative histogram
    hist_arr = np.bincount(orig_img.ravel())
    cum_hist_arr = np.cumsum(hist_arr / np.sum(hist_arr))

    # generate transformation lookup table
    transform_lut = np.floor(255 * cum_hist_arr).astype(np.uint8)

    # perform lookups and return resulting histogram equalized image
    return transform_lut[orig_img]
