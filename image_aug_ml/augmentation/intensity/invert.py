import numpy as np


def invert(orig_img: np.ndarray) -> np.ndarray:
    """Inverts image intensities.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to invert

    Returns
    -------
    np.ndarray
        inverted image
    """
    # if image isn't uint8, convert to it
    if not orig_img.dtype == np.uint8:
        # normalize image
        orig_img /= orig_img.max()

        # scale to 255 and convert to uint8
        orig_img = (255 * orig_img).astype(np.uint8)

    # invert image intensities and return
    return 255 - orig_img
