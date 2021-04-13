import numpy as np


def affine_transform(orig_img: np.ndarray, transform_mat: np.ndarray) -> np.ndarray:
    """Performs affine transformation on image and returns result.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to transform
    transform_mat : np.ndarray
        transformation matrix for affine transform, 3x3 matrix

    Returns
    -------
    np.ndarray
        affine transformed image
    """
    # init transformed image
    transformed_img = np.zeros_like(orig_img)

    # map each pixel in the output to a pixel in the input
    for ii, jj in np.ndindex(orig_img.shape[:2]):
        x, y, _ = transform_mat @ np.array([ii, jj, 1])

        if 0 < round(x) < orig_img.shape[0] and 0 < round(y) < orig_img.shape[1]:
            transformed_img[ii][jj] = orig_img[round(x)][round(y)]

    # return transformed image
    return transformed_img
