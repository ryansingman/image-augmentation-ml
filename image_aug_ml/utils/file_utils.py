from typing import List
import pathlib


def get_all_train_images(
    image_dir: pathlib.Path = pathlib.Path("./images/"),
) -> List[pathlib.Path]:
    """Gets all training images from image directory.

    Parameters
    ----------
    image_dir : pathlib.Path, optional
        directory to get images from, by default pathlib.Path("./images/")

    Returns
    -------
    List[pathlib.Path]
        list of all training image paths
    """
    return [*image_dir.glob("*/train/**/*.JPEG")]


def get_all_test_images(
    image_dir: pathlib.Path = pathlib.Path("./images/"),
) -> List[pathlib.Path]:
    """Gets all validation/testing images from image directory.

    Parameters
    ----------
    image_dir : pathlib.Path, optional
        directory to get images from, by default pathlib.Path("./images/")

    Returns
    -------
    List[pathlib.Path]
        list of all validation/testing image paths
    """
    return [*image_dir.glob("*/val/**/*.JPEG")]
