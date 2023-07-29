import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from tensorflow import data
import tensorflow_datasets
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from keras import losses
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
"""
writing a 5-fold image classifier by implementing a simple CNN for flower images
"""

import pathlib

dataset_url = "http://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

# getting len of image set
image_count = len(list(data_dir.glob('*/*.jpg')))

# showing a sample image (remember to use .show()) when working in VScode
roses = list(data_dir.glob('roses/*'))
len(roses)
Image.open(str(roses[1])).show()

# hyperparameters
batch_size = 32
image_height = 180
image_width = 180

# train and test split
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=.2,
    subset='validation',
    seed=123,
    image_size=(image_height,image_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(image_height, image_width),
  batch_size=batch_size)

# performing a bit of preprocessing
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# standardizing the data from [0,1]
normalization_layer = Rescaling(1./255)

normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# can start writing Convolutional Neural Network
model = Sequential([
    Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    Conv2D(16,3,padding='same',activation='relu'),
    MaxPooling2D(),
    Conv2D(32,3,padding='same',activation='relu'),
    MaxPooling2D(),
    Conv2D(63,3,padding='same',activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5)
])

# compiling model 
model.compile(
    optimizer='adam',
    loss= SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# training model using 10 epochs
epochs=10
history = model.fit(
  train_dataset,
  validation_data=val_ds,
  epochs=epochs
)

predictions = model.predict(train_dataset)
print(predictions)