import tensorflow as tf
import urllib.request
import zipfile
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    # _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
)
file_name = "horse-or-human.zip"
training_dir = "horse-or-human/training/"
urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.ZipFile(file_name, "r")
zip_ref.extractall(training_dir)
zip_ref.close()

train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    training_dir, target_size=(300, 300), class_mode="binary"
)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics=["accuracy"],
)


validation_url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = "horse-or-human/validation/"
urllib.request.urlretrieve(validation_url, validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name, "r")
zip_ref.extractall(validation_dir)
zip_ref.close()

validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(300, 300), class_mode="binary"
)

history = model.fit_generator(
    train_generator, epochs=15, validation_data=validation_generator
)

real_path = "horse-or-human/real/2.jpg"
real_img = image.load_img(real_path, target_size=(300, 300))
x = image.img_to_array(real_img)
x = np.expand_dims(x, axis=0)
image_tensor = np.vstack([x])
classes = model.predict(image_tensor)
print(classes)
print(classes[0])
if classes[0] > 0.5:
    print(real_path + "is a human")
else:
    print(real_path + "is a horse")

# print(model.summary())
