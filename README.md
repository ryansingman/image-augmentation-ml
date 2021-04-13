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
