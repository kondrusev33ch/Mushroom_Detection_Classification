import classification_train_models as k_nn
import detection_detr_model as detr
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from glob import glob
from tqdm import tqdm

IMG_WIDTH = 600
IMG_HEIGHT = 600


def get_image(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32) / 255.0


def get_mushrooms(model, imgs: list) -> list:
    """Set for DETR model"""
    model.eval()
    every_mushroom = []
    transforms = A.Compose([A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
                            ToTensorV2(p=1.0)], p=1.0)

    with torch.no_grad():
        for img in tqdm(imgs):
            img = transforms(image=img)['image']
            outputs = model(img.unsqueeze(0))

            outputs = [{k: v for k, v in outputs.items()}]
            p_boxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
            p_boxes = [np.array(box).astype(np.int32) for box in
                       A.augmentations.bbox_utils.denormalize_bboxes(p_boxes, IMG_HEIGHT, IMG_WIDTH)]
            prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]

            sample = img.permute(1, 2, 0).numpy()
            for box, p in zip(p_boxes, prob):
                if p > 0.65:
                    every_mushroom.append(sample[box[1]:box[3] + box[1], box[0]:box[2] + box[0]])
    return every_mushroom


def get_results(models, img, clss):
    results = pd.DataFrame({'labels': clss})
    for name, model in models:
        if name == 'ResNet50':
            transforms = A.Compose([A.Resize(height=200, width=200, p=1.0)], p=1.0)
        else:
            transforms = A.Compose([A.Resize(height=160, width=160, p=1.0)], p=1.0)

        img = transforms(image=img)['image']
        prediction = model.predict(np.expand_dims(img, axis=0))
        results[name] = np.array(prediction[0] * 100, dtype=int)

    results['sum'] = results.sum(axis=1)
    results = results.sort_values(by=['sum'], ascending=False).iloc[:5, :]

    return results

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Get input images
    images = []
    for filepath in glob('data/test_images/*.jpg'):
        images.append(get_image(filepath))

    # Load detection model
    detr_model = detr.get_model()
    detr_model.load_state_dict(torch.load('saved_models/detr_best_46_a.pth'))

    # Get all mushrooms from images
    mushrooms = get_mushrooms(detr_model, images)

    # Get classes from dataframe
    train_df = pd.read_csv('data/Classification/train.csv')
    classes = train_df['mushroom'].unique()
    n_classes = len(classes)

    # Load classification models
    base_model_resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_shape=(200, 200, 3))
    resnet_model = k_nn.get_pretrained_model(base_model_resnet, (200, 200, 3), n_classes)
    resnet_model.load_weights('saved_models/ResNet50_weights.ckpt')

    base_model_mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(160, 160, 3))
    mobilenet = k_nn.get_pretrained_model(base_model_mobilenet, (160, 160, 3), n_classes)
    mobilenet.load_weights('saved_models/MobileNetV2_weights.ckpt')

    cls_models = [('ResNet50', resnet_model), ('MobileNetV2', mobilenet)]

    # Show results
    for mushroom in mushrooms:
        table = get_results(cls_models, mushroom, classes)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mushroom)
        plt.subplot(1, 2, 2)
        ax_table = plt.table(cellText=table.values, colLabels=table.columns,
                             loc='center', bbox=[0, 0, 1, 1])
        ax_table.auto_set_font_size(False)
        ax_table.set_fontsize(13)

        plt.tight_layout()

        plt.show()
