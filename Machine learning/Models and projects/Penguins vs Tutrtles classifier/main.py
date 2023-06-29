"""
The goal of this code is to implement an image classifier using various Python libraries.
We will draw from a dataset from kaggle.com where each training set is an image of a penguin,
or a turtle. We will be able to create a binary classifier by using principles of computer vision
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_cv
"""
First we want to process the dataframe and assign a category (0 or 1) to each photo
"""
BATCH_DIM = 8

IMAGE_SIZE =(256, 256)

CLASSES = {
    'turtle' : 0,
    'penguin' : 1,
}

train_directory = "/Users/benstager/Desktop/archive-4/train/train"
test_directory = "/Users/benstager/Desktop/archive-4/valid/valid"

train_paths = sorted(os.path.join(train_directory, file) for file in os.listdir(train_directory))
test_paths = sorted(os.path.join(test_directory, file) for file in os.listdir(test_directory))

def preprocess_annotations(path):
    df = pd.read_json(path)
    df['category_id'] = df['category_id'].replace(2, 0)
    return df

train_annotations = preprocess_annotations('/Users/benstager/Desktop/archive-4/train_annotations')
test_annotations = preprocess_annotations('/Users/benstager/Desktop/archive-4/valid_annotations')

"""
We move on to begin writing our model, by using keras to define a computer vision model.
"""

resizer = keras.layers.Resizing(
    *IMAGE_SIZE, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)

augmenter = keras.Sequential(
    layers= [
        resizer,
         keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
    ]
)

def get_image(path, label, bbox):
    file = tf.io.read_file(path)
    return tf.image.decode_jpeg(file), label, bbox

"""
Converting to image detection format
"""
def to_dict(image, label, bbox):
    # Convert to object detection format expected by Keras CV
    bounding_boxes = {
        "classes": [label],
        "boxes": [bbox],
    }
    return {"images": image, "bounding_boxes": bounding_boxes}

"""
Function to generate datasets that the computer vision model will be trained on
"""
def generate_dataset(image_paths, annotations, augment=False):
    return (
        tf.data.Dataset.from_tensor_slices((
            image_paths,
            annotations["category_id"],
            annotations["bbox"].to_list())
        )
        .map(get_image, num_parallel_calls=tf.data.AUTOTUNE)
        .map(to_dict, num_parallel_calls=tf.data.AUTOTUNE)
        .ragged_batch(BATCH_DIM)
        .map(lambda b: augmenter(b) if augment else resizer(b), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

train_dataset = generate_dataset(train_paths, train_annotations, augment=True)
test_dataset = generate_dataset(test_paths, test_annotations)

