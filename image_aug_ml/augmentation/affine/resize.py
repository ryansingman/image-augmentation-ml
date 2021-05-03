from typing import Tuple

import numpy as np

from .affine_transform import affine_transform


def resize(
    orig_img: np.ndarray,
    scale_x_range: Tuple[float, float] = (0.5, 2),
    scale_y_range: Tuple[float, float] = (0.5, 2),
) -> np.ndarray:
    """Resizes an image by scale in provided scale range.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to resize
    scale_x_range : Tuple[float, float], optional
        range of x scale factors, by default (0.5, 2)
    scale_y_range : Tuple[float, float], optional
        range of y scale factors, by default (0.5, 2)

    Returns
    -------
    np.ndarray
        resized image
    """
    # find scale factors
    scale_x = np.random.uniform(*scale_x_range)
    scale_y = np.random.uniform(*scale_y_range)

    # build resize matrix
    resize_mat = np.array(
        [
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1],
        ]
    )

    # resize and return image
    return affine_transform(orig_img, resize_mat)
