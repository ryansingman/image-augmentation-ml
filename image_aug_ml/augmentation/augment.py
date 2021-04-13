from typing import Callable, Dict, List
import importlib
import pathlib
import tqdm

import cv2

from image_aug_ml.utils import get_all_original_train_images, save_to_file

# TODO REMOVE
import matplotlib.pyplot as plt


def augment_images(augmentation_conf: Dict, image_dir: pathlib.Path):
    """Augments training images in image directory and saves to file.

    Parameters
    ----------
    augmentation_conf : Dict
        augmentation config dictionary, contains information on what augmentations to perform
    image_dir : pathlib.Path
        directory to load original training images from
    """
    # get all original training image paths
    train_img_paths: List[pathlib.Path] = get_all_original_train_images(image_dir)

    # import all image augmentation operations
    augment_ops: List[Callable] = []
    for augment_path in augmentation_conf["augmentations"]:
        augment_module, augment_name = augment_path.rsplit(".", maxsplit=1)
        augment_ops.append(
            getattr(importlib.import_module(augment_module), augment_name)
        )

    # iterate over training images
    for train_img_path in tqdm.tqdm(train_img_paths):
        # load image from file
        train_img = cv2.imread(str(train_img_path))

        plt.imshow(train_img)

        # perform each augmentation on image
        for augment_op in augment_ops:
            # augment image
            augment_img = augment_op(train_img)

            # TODO REMOVE
            plt.figure()
            plt.imshow(augment_img)
            plt.title(augment_op.__name__)

            # save augmented image to file
            save_to_file(augment_img, train_img_path, augment_op.__name__)

        plt.show()
