from typing import Dict, List, Tuple
import pathlib

from tensorflow import keras
import tensorflow as tf


class ImageClassifier:
    """Image classifier, samples images from the dataset,
    preprocesses them, trains, and evaluates."""

    def __init__(
        self,
        img_dir: pathlib.Path,
        augmentation_conf: Dict,
        classifier_conf: Dict,
        model_name: str,
        resume_epoch: int = 0,
    ):
        """Creates image classifier model, creates dataset.

        Parameters
        ----------
        img_dir : pathlib.Path
            path to image directory
        augmentation_conf : Dict
            augmentation config, contains information about which image augmentations to use
        classifier_conf : Dict
            classifier config, contains info on input shape, hidden layer sizes, etc.
        model_name : str
            name of model, used for saving, loading checkpoints
        resume_epoch : int
            epoch to resume training from, by default 0 (no resume)
        """
        # save classifier config
        self.classifier_init_conf = classifier_conf["initialization"]
        self.classifier_train_conf = classifier_conf["train"]
        self.model_name = model_name
        self.resume_epoch = resume_epoch

        # create image dataset
        self.train_dataset, self.validation_dataset = self.make_dataset(
            img_dir,
            **augmentation_conf,
            image_shape=self.classifier_init_conf["image_shape"],
        )

        # create image classifier model
        if resume_epoch != 0:
            self.model = keras.models.load_model(
                f"checkpoints/{self.model_name}/{resume_epoch:02d}.cpkt", compile=False
            )
        else:
            self.model = self.make_model(**self.classifier_init_conf)

        # compile model with binary cross entropy loss and Adam optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                self.classifier_train_conf["learning_rate"]
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self):
        """Trains classifier for given number of epochs."""
        # set up save callback
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"checkpoints/{self.model_name}/{{epoch:02d}}.cpkt",
                verbose=True,
            )
        ]

        # find steps per epoch
        num_imgs = tf.data.experimental.cardinality(self.train_dataset).numpy()
        num_epochs = self.classifier_train_conf["epochs"]
        num_steps_per_epoch = int(num_imgs / num_epochs)

        # train model
        self.model.fit(
            self.train_dataset,
            epochs=num_epochs,
            initial_epoch=self.resume_epoch,
            callbacks=callbacks,
            validation_data=self.validation_dataset,
            validation_freq=15,
            steps_per_epoch=num_steps_per_epoch,
            verbose=True,
        )

    def evaluate(self) -> Dict[str, float]:
        """Evaluates model on validation data and returns loss and metrics."""
        return self.model.evaluate(self.validation_dataset, return_dict=True)

    @staticmethod
    def make_dataset(
        img_dir: pathlib.Path, augmentations: List[str], image_shape: List[int]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Makes training, validation datasets.

        Parameters
        ----------
        img_dir : pathlib.Path
            directory to find images
        augmentations : List[str]
            list of image augmentations to include in dataset
        image_shape: List[int]
            shape of images as list of ints

        Returns
        -------
        Tuple[tf.data.Dataset, tf.data.Dataset]
            tuple of training, validation datasets
        """
        # load original training, validation datasets
        train_ds = keras.preprocessing.image_dataset_from_directory(
            str(img_dir.joinpath("original/train")),
            labels="inferred",
            image_size=image_shape,
            label_mode="categorical",
        )
        val_ds = keras.preprocessing.image_dataset_from_directory(
            str(img_dir.joinpath("original/val")),
            labels="inferred",
            image_size=image_shape,
            label_mode="categorical",
        )

        # create list of augmentations dirs
        augmentation_dirs = [
            aug_name.rsplit(".", maxsplit=1)[-1] for aug_name in augmentations
        ]

        # add augmented images to training dataset
        for aug_dir in augmentation_dirs:
            train_ds = train_ds.concatenate(
                keras.preprocessing.image_dataset_from_directory(
                    str(img_dir.joinpath(f"{aug_dir}/train")),
                    labels="inferred",
                    image_size=image_shape,
                    label_mode="categorical",
                )
            )

        # set up shuffling
        train_ds = train_ds.shuffle(
            tf.data.experimental.cardinality(train_ds).numpy(),
            reshuffle_each_iteration=True,
        )
        val_ds = val_ds.shuffle(
            tf.data.experimental.cardinality(val_ds).numpy(),
            reshuffle_each_iteration=True,
        )

        # set up prefetching
        train_ds = train_ds.prefetch(buffer_size=64)
        val_ds = val_ds.prefetch(buffer_size=64)

        # return datasets
        return train_ds, val_ds

    @staticmethod
    def make_model(
        image_shape: List[int], hl_sizes: List[int], num_classes: int
    ) -> keras.Model:
        """Makes keras model with given input shape, hidden layer sizes.

        Parameters
        ----------
        image_shape : List[int]
            shape of input images
        hl_sizes : List[int]
            list of hidden layer sizes
        num_classes: int
            number of image classes to output

        Returns
        -------
        keras.Model
            keras model to classify images with
        """
        # initialize sequential model
        model = keras.models.Sequential()

        # create entry block
        model.add(keras.layers.experimental.preprocessing.Rescaling(1.0 / 255))
        model.add(keras.layers.Conv2D(32, 3, strides=2, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Conv2D(64, 3, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        # create hidden layer block
        for size in hl_sizes:
            model.add(keras.layers.Activation("relu"))
            model.add(keras.layers.SeparableConv2D(size, 3, padding="same"))
            model.add(keras.layers.BatchNormalization())

            model.add(keras.layers.Activation("relu"))
            model.add(keras.layers.SeparableConv2D(size, 3, padding="same"))
            model.add(keras.layers.BatchNormalization())

            model.add(keras.layers.MaxPooling2D(3, strides=2, padding="same"))

        # create output layer block
        model.add(keras.layers.SeparableConv2D(1024, 3, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.GlobalAveragePooling2D())

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        # return completed model
        return model
