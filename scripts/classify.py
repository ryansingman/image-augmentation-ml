"""Augments all training images using specified aumentations."""
import pathlib

import yaml

from image_aug_ml.classifier import ImageClassifier


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Image Augmentation")
    parser.add_argument("augmentation_conf", help="path to augmentation config")
    parser.add_argument("classifier_conf", help="path to classifier config")
    parser.add_argument(
        "--image_dir", help="path to image directory", default="./images/"
    )

    args = parser.parse_args()

    # load augmentation dict from file
    with open(args.augmentation_conf, "r") as augmentation_conf_file:
        augmentation_dict = yaml.load(augmentation_conf_file, Loader=yaml.SafeLoader)

    # load classifier dict from file
    with open(args.classifier_conf, "r") as classifier_conf_file:
        classifier_dict = yaml.load(classifier_conf_file, Loader=yaml.SafeLoader)

    # initialize image classifier model and datasets
    classifier = ImageClassifier(
        pathlib.Path(args.image_dir), augmentation_dict, classifier_dict
    )

    # train image classifier
    model_name = pathlib.Path(args.augmentation_conf).name.split(".")[0]
    classifier.train(model_name)
