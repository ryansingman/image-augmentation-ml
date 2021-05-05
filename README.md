# image-augmentation-ml
Finding the optimal set of image augmentations for image classification.

## Installation Instructions
First, ensure that you have the latest build tools installed on your machine:
```
python3 -m pip install --upgrade build
python3 -m build
```

Then, you can install the `image-augmentation-ml` package as follows:
```
python3 -m pip install -e .
```

## Development Tools
Install the pre-commit checker by running the following:
```
python3 -m pip install pre-commit
pre-commit install
```

From this point forward, whenever you commit to this repository, an autoformatter (black) and a linter (flake8) will run. If either come up with errors, you must fix them, add them to the staging area, and commit again.

## Testing Instructions
You can run the test suite with the following command:
```
tox
```

## Imageset Download
You can download the imageset using the following command.
```
make img_download
```

To specify the image size (either 160px or 320px), add on the `PX` flag. By default, this command will download the 160px dataset.

```
make img_download PX=320
```

## Image Augmentation
The set of augmented images can be created by running the augment script. To create the entire set of augmented images, you can run the following command:
```
python scripts/augment.py configs/augmentation/all.yaml
```

## Image Classification
To train an image classifier, you can run the following command. This command will train an image classifier on the set of all augmented images:
```
python scripts/classify.py configs/augmentation/all.yaml configs/classifier/default.yaml
```

## Classifier Evaluation
To evaluate the results of a classifier, you can run the following command. This command will evaluate the 30th epoch of the classifier that trained on all augmented images:
```
python scripts/evaluate.py configs/augmentation/all.yaml configs/classifier/default.yaml --load_epoch 30
```

Then, to visualize the results, you can plot them as follows:
```
python scripts/plot_results.py
```
