import pathlib

import cv2
import numpy as np


def save_to_file(
    augment_img: np.ndarray, train_img_path: pathlib.Path, augment_name: str
):
    """Saves image to file, with path built using the train image path and the augmentation name.

    Parameters
    ----------
    augment_img : np.ndarray
        augmented image as numpy array
    train_img_path : pathlib.Path
        path where the original image was loaded from
    augment_name : str
        name of the augmentation operation performed
    """
    # create augmented image path
    aug_img_path = pathlib.Path(
        *[part if part != "original" else augment_name for part in train_img_path.parts]
    )

    # make directory if doesn't already exist
    aug_img_path.parent.mkdir(exist_ok=True)

    # save image to path
    cv2.imwrite(aug_img_path, augment_img)
