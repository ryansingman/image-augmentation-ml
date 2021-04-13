import numpy as np

from .affine_transform import affine_transform


def rotate(orig_img: np.ndarray, max_theta: float = 360) -> np.ndarray:
    """Rotates an image by up to max_theta.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to rotate
    max_theta : float, optional
        maximum positive rotation (in degrees), by default 360

    Returns
    -------
    np.ndarray
        rotated image array
    """
    # find rotation (in radians)
    theta = np.random.uniform(0, max_theta) * np.pi / 180.0

    # build rotation matrix
    rotate_mat = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # rotate and return image
    return affine_transform(orig_img, rotate_mat)
