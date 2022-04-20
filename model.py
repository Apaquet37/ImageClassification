import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import os

num_skipped = 0
for folder_name in ("Button", "Switch"):
    folder_path = os.path.join("Images", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)


image_size = (180, 180) #could change to 32x32? that's the cifar size
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Images",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Images",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
