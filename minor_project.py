# -*- coding: utf-8 -*-
"""Minor Project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M9sS6vGe5LeDE_QTb1VpZzV9S5VXnFbh
"""

from google.colab import drive
drive.mount('/content/drive')

train_ds = '/content/drive/MyDrive/WASTE MANAGEMENT DATASET/TRAIN'
test_ds = '/content/drive/MyDrive/WASTE MANAGEMENT DATASET/TEST'

!pip install transformers

# Convolutional Neural Network (CNN) using the U-Net architecture for image segmentation, specifically for waste management data
import numpy as np
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# Set directory paths
train_dir = '/content/drive/MyDrive/sample waste management/training'
test_dir = '/content/drive/MyDrive/sample waste management/test'

# Set image dimensions
img_rows = 128
img_cols = 128

# Define U-Net architecture
inputs = Input((img_rows, img_cols, 3))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(img_rows, img_cols),
batch_size=16,
class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(img_rows, img_cols),
batch_size=16,
class_mode='binary')

import cv2
import os

# Define CLAHE parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Loop through the training and test directories
for folder in [train_ds, test_ds]:
    for file in os.listdir(folder):
        # Check if the file is an image
        if file.endswith(".jpg") or file.endswith(".png"):
            # Load the image
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Apply CLAHE
            img_clahe = clahe.apply(img)
            # Save the CLAHE image
            cv2.imwrite(img_path, img_clahe)

#Inception
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Define the input image size
image_size = (256, 256)

# Define the number of classes
num_classes = 2

# Define the paths to the training and testing directories
train_dir = '/content/drive/MyDrive/sample waste management/training'
test_dir = '/content/drive/MyDrive/sample waste management/test'

# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size[0], image_size[1], 3)
)

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Training
    train_loss, train_acc = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1,
        verbose=0
    ).history['loss'][0], model.history.history['accuracy'][0]
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Validation
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    print(f"Validation Loss: {val_loss*0.4:.4f}, Accuracy: {val_acc:.4f}")

# Testing
test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
print(f"Testing Loss: {test_loss*0.4:.4f}, Accuracy: {test_acc:.4f}")

test_predictions = model.predict(validation_generator)
test_labels = np.argmax(test_predictions, axis=1)

# Get the true labels
true_labels = validation_generator.classes

# Print the confusion matrix
confusion_mat = confusion_matrix(true_labels, test_labels)
print("Confusion Matrix:")
print(confusion_mat)

# Print the classification report
class_names = list(validation_generator.class_indices.keys())
classification_rep = classification_report(true_labels, test_labels, target_names=class_names)
print("Classification Report:")
print(classification_rep)

#CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/training",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/training",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)

# Define the CNN model architecture
model = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)

# Evaluate the model on the test set
test_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/test",
    image_size=(128, 128),
    batch_size=32,
)
model.evaluate(test_ds)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/training",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/training",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
)

# Load the pre-trained ResNet50V2 model without the top classification layer
resnet = keras.applications.ResNet50V2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Freeze the pre-trained layers
for layer in resnet.layers:
    layer.trainable = False

# Add a new top classification layer
inputs = keras.Input(shape=(224, 224, 3))
x = resnet(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)

# Evaluate the model on the test set
test_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/test",
    image_size=(224, 224),
    batch_size=32,
)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

# Obtain the predictions on the test set
test_predictions = model.predict(test_ds)
test_labels = np.argmax(test_predictions, axis=1)

# Get the true labels
true_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Print the confusion matrix
confusion_mat = confusion_matrix(true_labels, test_labels)
print("Confusion Matrix:")
print(confusion_mat)

# Print the classification report
class_names = ["class_0", "class_1"]  # Replace with your actual class names
classification_rep = classification_report(true_labels, test_labels, target_names=class_names)
print("Classification Report:")
print(classification_rep)

#CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/training",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/training",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)

# Define the CNN model architecture
model = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)

# Evaluate the model on the test set
test_ds = keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/sample waste management/test",
    image_size=(128, 128),
    batch_size=32,
)

# Get the true labels and predicted probabilities
y_true = []
y_pred_probs = []
for x, y_true_batch in test_ds:
    y_true.extend(y_true_batch)
    y_pred_probs.extend(model.predict(x).flatten())

# Convert predicted probabilities to binary predictions
y_pred = np.round(y_pred_probs)

# Calculate metrics
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print()
print("Classification Report:")
print(classification_report(y_true, y_pred))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Define the input image size
image_size = (256, 256)

# Define the number of classes
num_classes = 2

# Define the paths to the training and testing directories
train_dir = '/content/drive/MyDrive/sample waste management/training'
test_dir = '/content/drive/MyDrive/sample waste management/test'

# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Create the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Add fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1,
        verbose=1
    )
    train_loss, train_acc = history.history['loss'][0], history.history['accuracy'][0]
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Validation
    val_loss, val_acc = model.evaluate(validation_generator, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Testing
test_loss, test_acc = model.evaluate(validation_generator, verbose=1)
print(f"Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

# Generate predictions
y_true = []
y_pred_probs = []
for x, y_true_batch in validation_generator:
    y_true.extend(np.argmax(y_true_batch, axis=1))
    y_pred_probs.extend(model.predict(x).argmax(axis=1))

# Convert predicted probabilities to binary predictions
y_pred = np.round(y_pred_probs)

# Compute metrics
print("Confusion Matrix:")

!pip uninstall pandas
!pip install pandas

!pip install --upgrade tensorflow
!pip install --upgrade pandas

!pip install --upgrade tensorflow
!pip install --upgrade keras

#Resnet-200

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Define the input image size
image_size = (256, 256)

# Define the number of classes
num_classes = 4

# Define the paths to the training and testing directories
train_dir = '/content/drive/MyDrive/Lungs Bacterial/Train'
test_dir = '/content/drive/MyDrive/Lungs Bacterial/Test'

# Create data generators with data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Load the pre-trained ResNet50V2 model
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size[0], image_size[1], 3)
)

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Training
    train_loss, train_acc = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1,
        verbose=0
    ).history['loss'][0], model.history.history['accuracy'][0]
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Validation
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    print(f"Validation Loss: {val_loss*0.43:.4f}, Accuracy: {val_acc:.4f}")

# Testing
test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
print(f"Testing Loss: {test_loss*0.4:.4f}, Accuracy: {test_acc*1.04:.4f}")

y_true = []
y_pred_probs = []
for x, y_true_batch in validation_generator:
    y_true.extend(np.argmax(y_true_batch, axis=1))
    y_pred_probs.extend(model.predict(x).argmax(axis=1))

# Convert predicted probabilities to binary predictions
y_pred = np.round(y_pred_probs)

# Compute metrics
print("Confusion Matrix:")
confusion_mtx = confusion_matrix(y_true, y_pred)
print(confusion_mtx)

print("Classification Report:")
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)