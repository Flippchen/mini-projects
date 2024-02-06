import os
import pathlib
import logging
from keras import layers
import keras
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_model_score(history, name: str) -> None:
    """
    Plots the accuracy and loss of the model

    :param history: History of the model
    :param name: Name of the training
    :return:
    """
    plot_save_path = f'./acc-loss-{name}-model.png'
    # Read history and plot model score
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(plot_save_path)


def load_dataset(path: str, batch_size: int, img_height: int, img_width: int, seed: int) -> tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """
    :param path: Path to the Dataset folder
    :param batch_size: Integer which defines how many Images are in one Batch
    :param img_height: Height of the images to be loaded with
    :param img_width: Width of the images to be loaded with
    :return: Tuple of train, val Dataset and Class names
    """
    data_dir = pathlib.Path(path)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        seed=seed,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        seed=seed,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def show_augmented_batch(train_ds, data_augmentation) -> None:
    """
    Shows a sample batch of augmented images

    :param train_ds: Dataset to show a sample batch from
    :param data_augmentation: Data augmentation layers
    :return:
    """
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


def suppress_tf_warnings():
    """
    Suppresses TensorFlow warnings

    :return:
    """
    # Suppress TensorFlow INFO and WARNING logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["KMP_AFFINITY"] = "noverbose"

    # Suppress Python logging warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Suppress any deprecated function warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def show_sample_batch(train_ds: tf.data.Dataset, class_names: list) -> None:
    """
    Plots a sample batch of images from the dataset

    :param train_ds: Dataset to show a sample batch from
    :param class_names: Class names of the dataset
    :return: None
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def show_batch_shape(train_ds: tf.data.Dataset) -> None:
    """
    Prints the shape of the first batch of the dataset

    :param train_ds: Dataset to show the shape of the first batch from
    :return:
    """
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def create_augmentation_layer(img_height: int, img_width: int) -> keras.Sequential:
    """
    Creates a data augmentation layer

    :param img_height: Height of the images to be loaded with
    :param img_width: Width of the images to be loaded with
    :return: Sequential layer with data augmentation
    """
    return keras.Sequential(
        [
            layers.RandomFlip("vertical",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.GaussianNoise(0.1)
        ]
    )
