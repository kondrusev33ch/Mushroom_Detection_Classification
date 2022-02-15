"""
Resources:
    DataGenerator:
        https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    Classification basics:
        https://www.tensorflow.org/tutorials/images/classification
    Models:
        https://www.kaggle.com/pestipeti/keras-cnn-starter/notebook
        https://analyticsindiamag.com/mobilenet-vs-resnet50-two-cnn-transfer-learning-light-frameworks/
"""

import pandas as pd
import numpy as np
import random
import cv2
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental import preprocessing

from classification_helpers import plot_raw_images, plot_training_curves

IMG_WIDTH = 160
IMG_HEIGHT = 160
BATCH_SIZE = 8
N_EPOCHS = 300


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Cause of the error in MobilNet model training process
    np.random.seed(seed)
    tf.random.set_seed(seed)


def one_image_test(img_path, model, clss):
    """Takes image as input and prints top 5 predictions as dataframe"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0

    prediction = model.predict(np.expand_dims(img, axis=0))

    results = pd.DataFrame({'label': clss, 'percent': prediction[0]})
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(results.sort_values(by=['percent'], ascending=False).iloc[:5, :])


# Data generator
# =====================================================================================================
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, l_map):
        self.dataset = dataset.copy().sample(frac=1.0).reset_index(drop=True)
        self.labels_map = l_map

        self.n_labels = self.dataset['mushroom'].nunique()

    def __len__(self):
        return len(self.dataset) // BATCH_SIZE

    def __get_output(self, label):
        """label = 3, self.n_labels = 5
           return array([0, 0, 0, 1, 0])"""
        return keras.utils.to_categorical(label, num_classes=self.n_labels)

    def __get_input(self, img_id):
        img = cv2.imread('data/Classification/train/' + img_id + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

        return img.astype(np.float32) / 255.0

    def __getitem__(self, index):
        batches = self.dataset[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

        images = np.asarray([self.__get_input(img_id) for img_id in batches['image_id']])
        labels = np.asarray([self.__get_output(self.labels_map[label]) for label in batches['mushroom']])

        return images, labels

    def on_epoch_end(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


# Models
# =====================================================================================================
def get_custom_model(augmentations, input_shape, num_classes):
    return models.Sequential([augmentations,
                              layers.Conv2D(64, 5, padding='same', activation='relu', input_shape=input_shape),
                              layers.BatchNormalization(axis=3),
                              layers.MaxPooling2D(2),

                              layers.Conv2D(128, 3, padding='same', activation='relu'),
                              layers.AveragePooling2D(3),

                              layers.Flatten(),
                              layers.Dense(500, activation="relu"),
                              layers.Dropout(0.5),
                              layers.Dense(num_classes, activation='softmax')])


def prepare_model_for_transfer_learning(base_model, num_classes, augmentations=None):
    model = models.Sequential()
    if augmentations:
        model.add(augmentations)

    model.add(base_model)
    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu', input_dim=512))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def get_pretrained_model(base_model, input_shape, num_classes, augmentations=None):
    pretrained_model = prepare_model_for_transfer_learning(base_model, num_classes, augmentations)
    pretrained_model.build((None, *input_shape))
    return pretrained_model


# Training
# =====================================================================================================
def run(model, t_generator, v_generator, callbacks, name):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    t_history = model.fit(t_generator,
                          validation_data=v_generator,
                          epochs=N_EPOCHS,
                          callbacks=callbacks)
    model.save_weights(f'{name}_weights.ckpt')

    return t_history


# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    seed_everything()

    train_df = pd.read_csv('data/Classification/train.csv')
    print(train_df)
    #     image_id mushroom
    # 0    00_beli     beli
    # 1    00_dozh     dozh
    # 2    00_dubo     dubo
    # 3    00_lisi     lisi
    # 4    00_masl     masl
    # ..       ...      ...
    # 514  49_dozh     dozh
    # 515  50_dozh     dozh
    # 516  51_dozh     dozh
    # 517  52_dozh     dozh
    # 518  53_dozh     dozh

    classes = train_df['mushroom'].unique()
    number_of_classes = len(classes)
    print('Number of mushroom classes:', number_of_classes)
    # => Number of mushroom classes: 15
    print(train_df.groupby('mushroom').count())
    #           image_id
    # mushroom
    # beli            36
    # dozh            54
    # dubo            31
    # lisi            31
    # masl            34
    # mayr            31
    # muho            36
    # pant            47
    # podb            41
    # poga            31
    # prek            44
    # ryad            33
    # siro            37
    # svin            35
    # zhel            32

    # Preprocessing
    # =================================================================================================
    # validation dataset = 15% from each training dataset class
    valid_df = pd.DataFrame()
    for cls in train_df['mushroom'].unique():
        df = train_df[train_df['mushroom'] == cls]
        valid_df = valid_df.append(df.sample(frac=0.15))

    valid_df = valid_df.sort_index()
    train_df = train_df.drop(valid_df.index)

    # Plot 4 random images from train dataset
    plot_raw_images(train_df)
    # All images have different sizes and, they are pretty big

    # We need to convert mushroom classes to numerical values for training
    labels_map = dict(zip(classes, range(number_of_classes)))

    # I will use custom data generator
    train_generator = DataGenerator(train_df, labels_map)
    valid_generator = DataGenerator(valid_df, labels_map)

    # Set arguments and train
    # =================================================================================================
    data_augmentation = keras.Sequential([preprocessing.RandomFlip('horizontal'),
                                          preprocessing.RandomRotation(0.3),
                                          preprocessing.RandomZoom(0.2)])

    callbacks = [keras.callbacks.EarlyStopping(patience=10,
                                               monitor='val_accuracy',
                                               restore_best_weights=True)]
    in_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    # Note: I trained one model at a time!

    # CustomCNN
    # =========
    # Variables I used: BATCH_SIZE = 32, IMG_SIZE = 200x200 [!!!]
    # customnet = get_custom_model(data_augmentation, (IMG_HEIGHT, IMG_WIDTH, 3), number_of_classes)
    # customnet_history = run(customnet, train_generator, valid_generator, callbacks, 'CustomNet')
    # plot_training_curves(customnet_history)

    # ResNet50
    # ========
    # Variables I used: BATCH_SIZE = 8, IMG_SIZE = 200x200 [!!!]
    # base_model_resnet = tf.keras.applications.resnet50.ResNet50(
    #     include_top=False, weights='imagenet', input_shape=in_shape)
    # resnet = get_pretrained_model(base_model_resnet, in_shape, number_of_classes, data_augmentation)
    # resnet_history = run(resnet, train_generator, valid_generator, callbacks, 'ResNet50')
    # plot_training_curves(resnet_history)

    # MobileNetV2
    # ===========
    # Variables I used: BATCH_SIZE = 8, IMG_SIZE = 160x160 [!!!]
    base_model_mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=in_shape)
    mobilenet = get_pretrained_model(base_model_mobilenet, in_shape, number_of_classes, data_augmentation)
    mobilenet_history = run(mobilenet, train_generator, valid_generator, callbacks, 'MobileNetV2')
    plot_training_curves(mobilenet_history)

    # Test
    # =================================================================================================
    # my_model = get_custom_model(data_augmentation, (IMG_HEIGHT, IMG_WIDTH, 3), number_of_classes)
    # my_model.load_weights('saved_models/custom_nn_weights.ckpt')
    # one_image_test('data/Classification/test_images/test_image_beli.jpg', my_model, classes)
    #
    # resnet = get_pretrained_model(base_model_resnet, in_shape, number_of_classes, data_augmentation)
    # resnet.load_weights('saved_models/ResNet50_weights.ckpt')
    # one_image_test('data/Classification/test_images/test_image_beli.jpg', resnet, classes)

    mobilenet = get_pretrained_model(base_model_mobilenet, in_shape, number_of_classes, data_augmentation)
    mobilenet.load_weights('saved_models/MobileNetV2_weights.ckpt')
    one_image_test('data/Classification/test_images/test_image_beli.jpg', mobilenet, classes)
