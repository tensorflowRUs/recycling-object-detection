import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import datetime
from pathlib import Path


def load_data():
    my_path = os.path.abspath(os.path.dirname(__file__))
    base = Path(my_path).parent

    logs_base_dir = os.path.join(base, "logs")
    logs_dir_fit = os.path.join(logs_base_dir, "fit")
    directory = os.path.join(logs_dir_fit, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(directory).mkdir(parents=True, exist_ok=True)


    tensorboard_callback = TensorBoard(
        log_dir=directory, update_freq='batch', histogram_freq=1)

    base_dir = os.path.join(base, "data")
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    test_dir = os.path.join(base_dir, 'test')

    path = os.path.join(os.path.join(test_dir, 'compost'), 'banana1.jpg')
    abs_path = os.path.abspath(path)
    img = image.load_img(abs_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # train_compost_dir = os.path.join(train_dir, 'compost')
    # validation_compost_dir = os.path.join(validation_dir, 'compost')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  # optimizer=RMSprop(lr=1e-4),
                  optimizer='adam',
                  metrics=['accuracy'])

    train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

    validation_data_gen = ImageDataGenerator(rescale=1. / 255,
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             fill_mode='nearest')

    train_generator = train_data_gen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        # batch_size=20,
        class_mode='categorical')

    validation_generator = validation_data_gen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        # batch_size=20,
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1,  # 2000 images = batch_size * steps
        epochs=100,
        validation_data=validation_generator,
        validation_steps=1,  # 1000 images = batch_size * steps
        verbose=1,
        callbacks=[tensorboard_callback])

    classes = model.predict(images)
    print(classes)


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Starting...")
    load_data()


if __name__ == "__main__":
    main()
