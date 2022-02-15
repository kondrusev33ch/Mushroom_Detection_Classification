import pandas as pd
import cv2
import matplotlib.pyplot as plt


def plot_raw_images(dataset: pd.DataFrame):
    images = []
    for img_id in dataset['image_id'].sample(4):
        img = cv2.imread('data/Classification/train/' + img_id + '.jpg')
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()
    for image, ax in zip(images, axes):
        ax.imshow(image)
    plt.show()


def plot_training_curves(train_history):
    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    pass
