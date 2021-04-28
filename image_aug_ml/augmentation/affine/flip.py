import numpy as np

from .affine_transform import affine_transform


def flip_vertical(orig_img: np.ndarray) -> np.ndarray:
    """Flips an image vertically.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to flip vertically

    Returns
    -------
    np.ndarray
        vertically flipped image
    """
    # find height
    img_height = orig_img.shape[0]

    # build vertical flip matrix
    vert_flip_mat = np.array([[-1, 0, img_height - 1], [0, 1, 0], [0, 0, 1]])

    # flip vertically and return image
    return affine_transform(orig_img, vert_flip_mat)


def flip_horizontal(orig_img: np.ndarray) -> np.ndarray:
    """Flips an image horizontally.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to flip horizontally

    Returns
    -------
    np.ndarray
        horizontally flipped image
    """
    # find width
    img_width = orig_img.shape[1]

    # build horizontal flip matrix
    horiz_flip_mat = np.array([[1, 0, 0], [0, -1, img_width - 1], [0, 0, 1]])

    # flip horizontally and return image
    return affine_transform(orig_img, horiz_flip_mat)
