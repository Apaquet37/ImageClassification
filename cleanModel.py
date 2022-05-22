import tensorflow as tf #This set of lines is standard syntax for importing ML libraries
from tensorflow import keras
from tensorflow.keras import layers


import os

input_shape = (32,)

image_size = (180, 180) #The image size, this could be changed to something smaller or bigger
batch_size = 32 #Run through 32 images at a time

train_ds = tf.keras.preprocessing.image_dataset_from_directory( #Importing the images from the computer
    "Images", #Name of directory that contains the images on the computer, I have two folders within it, one named Button and one named Switch
    validation_split=0.2, #20% of the images will be saved for validation, 80% are used for training
    subset="training", #Naming this 80% the training dataset
    seed=713, #A random number, standard throughout the project that ensures that the random way the data splits is the same way each time
    image_size=image_size, #Setting image and batch size
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory( #Same process but for validation data
    "Images",
    validation_split=0.2,
    subset="validation",
    seed=713,
    image_size=image_size,
    batch_size=batch_size,
)

        
#data augmentation - transformations made to the images that still preserves what they are
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"), #A horizontal flip - a dog is still a dog, whether you flip it horizontally or not
        layers.RandomRotation(0.1), #Random rotations to add variance to data
    ]
)

        
#standardize the data (this is being done as part of the model, with cpu it might be better to do it beforehand)
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x) #Rescaling the RGB values so they're from 0-1 instead of 0-255

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


#Making the Model

def make_model(input_shape, num_classes): #Most ML models are just blocks of code repeating to form layers
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x) 
    activation = "sigmoid" #Setting the type of dense activation layer - sigmoid returns a zero or one
    units = 1

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x) #Finally making a decision
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
