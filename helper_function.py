import tensorflow as tf
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import zipfile
import datetime

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes it into
    (img_shape, img_shape, 3). Optionally scales the image to range(0, 1).

    Parameters:
    - filename (str): Filename of the target image.
    - img_shape (int): Size to resize target image to, default 224.
    - scale (bool): Whether to scale pixel values to range(0, 1), default True.

    Returns:
    - Tensor: The preprocessed image tensor.
    """
    if not os.path.isfile(filename):
        raise ValueError(f"File {filename} does not exist.")
    
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    
    if scale:
        img = img / 255.0
    
    return img


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """
    Creates a labelled confusion matrix comparing predictions and ground truth labels.

    Parameters:
    - y_true: Array of ground truth labels.
    - y_pred: Array of predicted labels.
    - classes: Array of class labels, if None, integer labels are used.
    - figsize: Size of the output figure.
    - text_size: Size of the output text.
    - norm: Normalize values or not.
    - savefig: Save confusion matrix to file.

    Returns:
    - A labelled confusion matrix plot comparing y_true and y_pred.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if norm else cm
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(n_classes)
    ax.set(title="Confusion Matrix", xlabel="Predicted label", ylabel="True label",
           xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=labels, yticklabels=labels)
    
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(n_classes), range(n_classes)):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)" if norm else f"{cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
    
    if savefig:
        fig.savefig("confusion_matrix.png")

    plt.show()

def pred_and_plot(model, filename, class_names):
    """
    Imports an image, makes a prediction with a model, and plots the image with the predicted class.

    Parameters:
    - model: Trained TensorFlow model.
    - filename: Path to the image file.
    - class_names: List of class names.

    Returns:
    - None: Displays the image with prediction as title.
    """
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    pred_class = class_names[tf.argmax(pred[0])] if len(pred[0]) > 1 else class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instance to store log files.

    Parameters:
    - dir_name: Directory to store TensorBoard log files.
    - experiment_name: Name of the experiment directory.

    Returns:
    - TensorBoard callback instance.
    """
    if not isinstance(dir_name, str) or not isinstance(experiment_name, str):
        raise ValueError("dir_name and experiment_name must be strings.")
    
    log_dir = f"{dir_name}/{experiment_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Saving TensorBoard log files to: {log_dir}")
    
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def plot_loss_curves(history):
    """
    Plots loss and accuracy curves for training and validation data.

    Parameters:
    - history: TensorFlow model History object.

    Returns:
    - None: Displays the plots.
    """
    metrics = ['loss', 'accuracy']
    epochs = range(len(history.history['loss']))
    
    for metric in metrics:
        plt.plot(epochs, history.history[metric], label=f'training_{metric}')
        plt.plot(epochs, history.history[f'val_{metric}'], label=f'val_{metric}')
        plt.title(metric.capitalize())
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Parameters:
    - original_history: History object from original model training.
    - new_history: History object from continued model training.
    - initial_epochs: Number of epochs in original_history.
    
    Returns:
    - None: Displays the plots.
    """
    metrics = ['accuracy', 'loss']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 1, i+1)
        total_metric = original_history.history[metric] + new_history.history[metric]
        total_val_metric = original_history.history[f'val_{metric}'] + new_history.history[f'val_{metric}']
        
        plt.plot(total_metric, label=f'Training {metric.capitalize()}')
        plt.plot(total_val_metric, label=f'Validation {metric.capitalize()}')
        plt.axvline(x=initial_epochs-1, color='r', linestyle='--', label='Start Fine Tuning')
        plt.legend(loc='lower right' if metric == 'accuracy' else 'upper right')
        plt.title(f'Training and Validation {metric.capitalize()}')

    plt.xlabel('Epochs')
    plt.show()

def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Parameters:
    - filename (str): A filepath to a target zip folder to be unzipped.
    """
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Parameters:
    - dir_path (str): Target directory.

    Returns:
    - List of tuples: (dirpath, number of subdirectories, number of images in each subdirectory).
    """
    dir_structure = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
        dir_structure.append((dirpath, len(dirnames), len(filenames)))
    
    return dir_structure

def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and F1 score of a classification model.

    Parameters:
    - y_true: True labels (1D array).
    - y_pred: Predicted labels (1D array).

    Returns:
    - Dictionary: Accuracy, precision, recall, F1 score.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }