"""Augments all training images using specified aumentations."""
import yaml

from image_aug_ml.augmentation import augment_images


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Image Augmentation")
    parser.add_argument("augmentation_conf", help="path to augmentation config")
    parser.add_argument(
        "--image_dir", help="path to image directory", default="./images/"
    )

    args = parser.parse_args()

    # load augmentation dict from file
    with open(args.augmentation_conf, "r") as augmentation_conf_file:
        augmentation_dict = yaml.load(augmentation_conf_file)

    # augment images and save copies to filesystem
    augment_images(augmentation_dict, args.image_dir)
