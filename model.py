import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import os

input_shape = (32,)

num_skipped = 0
#for folder_name in ("Button", "Switch"):
    #folder_path = os.path.join("Images", folder_name)
    #for fname in os.listdir(folder_path):
        #fpath = os.path.join(folder_path, fname)
        #try:
            #fobj = open(fpath, "rb")
            #is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        #finally:
            #fobj.close()

        #if not is_jfif:
            #num_skipped += 1
            # Delete corrupted image
            #os.remove(fpath)

print("Deleted %d images" % num_skipped)


image_size = (180, 180) #could change to 32x32? that's the cifar size
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Images",
    validation_split=0.2,
    subset="training",
    seed=713,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Images",
    validation_split=0.2,
    subset="validation",
    seed=713,
    image_size=image_size,
    batch_size=batch_size,
)

#visualize data - is this necessary?
#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
    #for i in range(9):
        #ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(images[i].numpy().astype("uint8"))
        #plt.title(int(labels[i]))
        #plt.axis("off")
        
#data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

#visualizing augmentation - is this necessary?
#plt.figure(figsize=(10, 10))
#for images, _ in train_ds.take(1):
    #for i in range(9):
        #augmented_images = data_augmentation(images)
        #ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(augmented_images[0].numpy().astype("uint8"))
        #plt.axis("off")
        
#standardize the data (this is being done as part of the model, with cpu it might be better to do it beforehand)
#inputs = keras.Input(shape=input_shape)
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


#Making the Model

def make_model(input_shape, num_classes):
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
    if num_classes == 2: #once I decide for sure how many classes I'll have this can come out, but the syntax is useful while still in development
        activation = "sigmoid" #sigmoid returns a zero or one
        units = 1
    else:
        activation = "softmax" #softmax returns a probability for each class
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
