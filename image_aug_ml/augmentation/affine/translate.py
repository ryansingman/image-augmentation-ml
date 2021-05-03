import numpy as np

from .affine_transform import affine_transform


def translate(
    orig_img: np.ndarray, max_tx: float = 0.3, max_ty: float = 0.3
) -> np.ndarray:
    """Translates an image by up to max_tx, max_ty.

    Parameters
    ----------
    orig_img : np.ndarray
        original image to translate
    max_tx : float, optional
        maximum horizontal distance to translate image (proportion of pixels), by default 0.3
    max_ty : float, optional
        maximum vertical distance to translate image (proportion of pixels), by default 0.3

    Returns
    -------
    np.ndarray
        translated image array
    """
    # find translation (in pixels)
    tx_pix = orig_img.shape[0] * np.random.uniform(-max_tx, max_tx)
    ty_pix = orig_img.shape[0] * np.random.uniform(-max_ty, max_ty)

    # build translation matrix
    translate_mat = np.array(
        [
            [1, 0, tx_pix],
            [0, 1, ty_pix],
            [0, 0, 1],
        ]
    )

    # translate and return image
    return affine_transform(orig_img, translate_mat)
