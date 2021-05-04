"""Trains image classifier using set of augmented images."""
import pathlib

import yaml

from image_aug_ml.classifier import ImageClassifier


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Image classifier training")
    parser.add_argument("augmentation_conf", help="path to augmentation config")
    parser.add_argument("classifier_conf", help="path to classifier config")
    parser.add_argument(
        "--image_dir", help="path to image directory", default="./images/"
    )
    parser.add_argument(
        "--resume_epoch", help="epoch to resume training from", type=int, default=0
    )

    args = parser.parse_args()

    # load augmentation dict from file
    with open(args.augmentation_conf, "r") as augmentation_conf_file:
        augmentation_dict = yaml.load(augmentation_conf_file, Loader=yaml.SafeLoader)

    # load classifier dict from file
    with open(args.classifier_conf, "r") as classifier_conf_file:
        classifier_dict = yaml.load(classifier_conf_file, Loader=yaml.SafeLoader)

    # initialize image classifier model and datasets
    model_name = pathlib.Path(args.augmentation_conf).name.split(".")[0]
    classifier = ImageClassifier(
        pathlib.Path(args.image_dir),
        augmentation_dict,
        classifier_dict,
        model_name,
        args.resume_epoch,
    )

    # train image classifier
    classifier.train()
