import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import datetime
from pathlib import Path


def get_model():
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

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_log_dir():
    my_path = os.path.abspath(os.path.dirname(__file__))
    base = Path(my_path).parent
    logs_base_dir = os.path.join(base, "logs")
    logs_dir_fit = os.path.join(logs_base_dir, "fit")
    directory = os.path.join(logs_dir_fit, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(directory).mkdir(parents=True, exist_ok=True)

    return directory


def run(model, log_dir, mode='default', epochs=100):
    my_path = os.path.abspath(os.path.dirname(__file__))
    base = Path(my_path).parent

    tensorboard_callback = TensorBoard(
        log_dir=log_dir, update_freq='batch', histogram_freq=1)

    if mode == 'default':
        base_dir = os.path.join(base, "data")
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

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

        fit_model = model.fit_generator(
            train_generator,
            steps_per_epoch=1,  # 2000 images = batch_size * steps
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=1,  # 1000 images = batch_size * steps
            verbose=1,
            callbacks=[tensorboard_callback])

        return fit_model


def get_test_dir():
    my_path = os.path.abspath(os.path.dirname(__file__))
    base = Path(my_path).parent
    base_dir = os.path.join(base, "data")
    test_dir = os.path.join(base_dir, 'test')

    return test_dir


def get_test_image(test_dir, category='compost', file_name='banana1.jpg'):
    path = os.path.join(os.path.join(test_dir, category), file_name)
    abs_path = os.path.abspath(path)
    raw_img = image.load_img(abs_path, target_size=(150, 150))
    x = image.img_to_array(raw_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    return images, raw_img


def predict(model, test_image):
    classes = model.predict(test_image)
    print(classes)


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Starting...")
    model = get_model()
    log_dir = get_log_dir()
    fit_model = run(model, log_dir, mode='default')
    test_image = get_test_image(get_test_dir())
    predict(model, test_image)


if __name__ == "__main__":
    main()
