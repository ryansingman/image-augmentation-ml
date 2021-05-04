"""Evaluates image classifier using validation data."""
import pathlib

import yaml

from image_aug_ml.classifier import ImageClassifier


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Image classifier evaluation")
    parser.add_argument("augmentation_conf", help="path to augmentation config")
    parser.add_argument("classifier_conf", help="path to classifier config")
    parser.add_argument(
        "--image_dir", help="path to image directory", default="./images/"
    )
    parser.add_argument("--load_epoch", help="epoch to evaluate", type=int, default=0)

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
        args.load_epoch,
    )

    # evaluate image classifier
    classifier_results = classifier.evaluate()

    # save results to file
    with open(f"results/{model_name}_{args.load_epoch}.yaml", "w") as results_file:
        yaml.dump(classifier_results, results_file)
