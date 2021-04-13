import numpy as np

from .affine_transform import affine_transform


# vertical flip transformation matrix
vert_flip_mat: np.ndarray = np.array(
    [
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ]
)


# horizontal flip transformation matrix
horiz_flip_mat: np.ndarray = np.array(
    [
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]
)


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
    # flip vertically and return image
    return affine_transform(
        np.pad(
            orig_img, pad_width=((orig_img.shape[1], orig_img.shape[1]), (0, 0), (0, 0))
        ),
        vert_flip_mat,
    )[: orig_img.shape[1], :]


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
    # flip horizontally and return image
    return affine_transform(orig_img, horiz_flip_mat)
