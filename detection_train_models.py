import os
import cv2
import glob
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import detection_detr_model as detr
import detection_frcnn_model as frcnn

N_FOLDS = 5
SEED = 42
DIR_TRAIN = 'data/Detection/preprocessed/'
IMG_WIDTH = 200
IMG_HEIGHT = 200


# Seed everything
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Augmentations
# ======================================================================================================
def get_train_transforms(bb_format='pascal_voc'):
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                                    val_shift_limit=0.2, p=0.9),
                               A.RandomBrightnessContrast(brightness_limit=0.2,
                                                          contrast_limit=0.2, p=0.9)], p=0.9),
                      A.HorizontalFlip(p=0.5),
                      A.Cutout(num_holes=7, max_h_size=15, max_w_size=15, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                     p=1.0,
                     bbox_params=A.BboxParams(format=bb_format, label_fields=['labels']))


def get_valid_transforms(bb_format='pascal_voc'):
    return A.Compose([ToTensorV2(p=1.0)],
                     p=1.0,
                     bbox_params=A.BboxParams(format=bb_format, label_fields=['labels']))


# Creating Dataset
# ======================================================================================================
class MushroomDataset(Dataset):
    def __init__(self, image_ids, dataframe, bb_format='pascal_voc', transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        self.bb_format = bb_format

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['img_id'] == image_id]
        image = cv2.imread(f'{DIR_TRAIN}{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['box_x', 'box_y', 'box_w', 'box_h']].values

        # Area of bboxes
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = np.zeros(len(boxes), dtype=np.int32)
        if self.bb_format == 'pascal_voc':
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            labels = torch.ones((records.shape[0],), dtype=torch.int64)

        if self.transforms:
            sample = {'image': image,
                      'bboxes': boxes,
                      'labels': labels}

            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        # Normalizing bounding boxes
        if self.bb_format == 'coco':
            boxes = A.augmentations.bbox_utils.normalize_bboxes(boxes, rows=IMG_HEIGHT, cols=IMG_WIDTH)

        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                  'labels': torch.as_tensor(labels, dtype=torch.long),
                  'image_id': torch.tensor([index]),
                  'area': area}

        if self.bb_format == 'pascal_voc':
            target['is_crowd'] = torch.zeros((records.shape[0],), dtype=torch.int64)

        return image, target, image_id


# Get data
# ======================================================================================================
def collate_fn(batch):
    return tuple(zip(*batch))


def get_tv_data_loaders(fold, df_folds_, annotations_, bb_format='coco', batch_size=2):
    df_train = df_folds_[df_folds_['fold'] != fold]
    df_valid = df_folds_[df_folds_['fold'] == fold]

    train_dataset = MushroomDataset(image_ids=df_train.index.values,
                                    dataframe=annotations_,
                                    bb_format=bb_format,
                                    transforms=get_train_transforms(bb_format))
    valid_dataset = MushroomDataset(image_ids=df_valid.index.values,
                                    dataframe=annotations_,
                                    bb_format=bb_format,
                                    transforms=get_valid_transforms(bb_format))

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_fn)

    return train_data_loader, valid_data_loader


# Training
# ======================================================================================================
def run_detr_training(train_data_loader, valid_data_loader):
    model, train_losses, valid_losses = detr.run(train_data_loader, valid_data_loader, 50, 8)
    return model, train_losses, valid_losses


def run_frcnn_training(train_data_loader, valid_data_loader):
    model, train_losses, valid_precisions = frcnn.run(train_data_loader, valid_data_loader, 10, 2)
    return model, train_losses, valid_precisions


# Sample
# ======================================================================================================
def show_annotations_boxes(train_data_loader, bb_format='pascal_voc'):
    images, targets, images_ids = next(iter(train_data_loader))

    targets = [{k: v for k, v in t.items()} for t in targets]

    arr_boxes = [target['boxes'].cpu().numpy() for target in targets]
    if bb_format == 'coco':
        arr_boxes = [[np.array(box).astype(np.int32) for box in
                      A.augmentations.bbox_utils.denormalize_bboxes(boxes, IMG_HEIGHT, IMG_WIDTH)]
                     for boxes in arr_boxes]

    samples = [image.permute(1, 2, 0).cpu().numpy() for image in images]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for sample, boxes, ax in zip(samples, arr_boxes, axes):
        for box in boxes:
            if bb_format == 'pascal_voc':
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (220, 0, 0), 2)
            else:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2] + box[0], box[3] + box[1]),
                              (220, 0, 0), 2)
        ax.imshow(sample)
    plt.show()


def visual_test_on_single_image(img, model, device_, ax, model_name):
    test_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    test_image /= 255.0

    transforms = A.Compose([A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
                            ToTensorV2(p=1.0)], p=1.0)
    test_image = transforms(image=test_image)['image']

    test_image = test_image.to(device_)
    sample = test_image.permute(1, 2, 0).cpu().numpy()

    model.eval()
    model.to(device_)
    cpu_device = torch.device('cpu')

    with torch.no_grad():
        outputs = model(test_image.unsqueeze(0))

    if model_name == 'DETR':
        outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
        p_boxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
        p_boxes = [np.array(box).astype(np.int32) for box in
                   A.augmentations.bbox_utils.denormalize_bboxes(p_boxes, IMG_HEIGHT, IMG_WIDTH)]
        prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
    else:
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        p_boxes = outputs[0]['boxes'].data.cpu().numpy()
        prob = outputs[0]['scores'].data.cpu().numpy()

    for box, p in zip(p_boxes, prob):
        if model_name == 'FRCNN':
            if p > 0.8:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (220, 0, 220), 2)
        else:
            if p > 0.6:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2] + box[0], box[3] + box[1]),
                              (220, 0, 220), 2)

    ax.set_title(model_name)
    ax.imshow(sample)


# ------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    seed_everything(SEED)

    annotations = pd.read_csv('data/Detection/train_0000-1017.csv')
    print(tabulate(annotations.head(10), headers='keys'))
    #     label       box_x    box_y    box_w    box_h  img_id      img_width    img_height
    # --  --------  -------  -------  -------  -------  --------  -----------  ------------
    #  0  mushroom       33       38       78      103  0000.jpg          200           200
    #  1  mushroom       93       56       70      104  0000.jpg          200           200
    #  2  mushroom        6       29      190      130  0001.jpg          200           200
    #  3  mushroom       12        3      182      192  0002.jpg          200           200
    #  4  mushroom       52       27       82      123  0003.jpg          200           200
    #  5  mushroom       84       28       89       92  0004.jpg          200           200
    #  6  mushroom       19       64      154       91  0005.jpg          200           200
    #  7  mushroom       63       87       81       58  0006.jpg          200           200
    #  8  mushroom        5       24      182      169  0007.jpg          200           200
    #  9  mushroom       76       42       63      106  0008.jpg          200           200

    # Creating Folds
    # ==============
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    df_folds = annotations[['img_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1  # each row has 1 bounding box
    df_folds = df_folds.groupby('img_id').count()  # set how many bboxes on each image
    df_folds.loc[:, 'stratify_group'] = 'mushroom'

    for fold_number, (train_index, val_index) in enumerate(
            skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    print(df_folds)
    #           bbox_count stratify_group  fold
    # img_id
    # 0000.jpg           2       mushroom   2.0
    # 0001.jpg           1       mushroom   4.0
    # 0002.jpg           1       mushroom   2.0
    # 0003.jpg           1       mushroom   0.0
    # 0004.jpg           1       mushroom   0.0
    # ...              ...            ...   ...
    # 1013.jpg           3       mushroom   0.0
    # 1014.jpg           1       mushroom   1.0
    # 1015.jpg           2       mushroom   4.0
    # 1016.jpg           1       mushroom   2.0
    # 1017.jpg           1       mushroom   0.0

    # Train models
    # ============
    dt_data_loader, dv_data_loader = \
        get_tv_data_loaders(0, df_folds, annotations, bb_format='coco', batch_size=4)
    show_annotations_boxes(dt_data_loader, 'coco')
    # Time: 1 epoch cost me about 0.55 min on my computer
    detr_model, dt_losses, dv_losses = run_detr_training(dt_data_loader, dv_data_loader)

    # detr_model = detr.get_model()
    # detr_model.load_state_dict(torch.load('saved_models/detr_best_46.pth'))

    ft_data_loader, fv_data_loader = \
        get_tv_data_loaders(0, df_folds, annotations, bb_format='pascal_voc', batch_size=2)
    show_annotations_boxes(ft_data_loader, 'pascal_voc')
    # Time: 1 epoch cost me about 6.5 min on my computer
    frcnn_model, ft_losses, fv_precisions = run_frcnn_training(ft_data_loader, fv_data_loader)

    # frcnn_model = frcnn.get_model()
    # frcnn_model.load_state_dict(torch.load('saved_models/frcnn_best_9.pth'))

    # Plot DETR and FRCNN losses
    # ==========================
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(dt_losses, color='b', linewidth=2, label='train_loss')
    axes[0].plot(dv_losses, color='r', linewidth=2, label='valid_loss')
    axes[0].set_title('DETR')
    axes[0].legend()

    axes[1].plot(ft_losses, color='b', linewidth=2, label='train_loss')
    axes[1].plot(fv_precisions, color='g', linewidth=2, label='valid_precision')
    axes[1].set_title('FRCNN')
    axes[1].legend()

    plt.show()

    # Visual check
    # ============
    test_images = []
    for filename in glob.glob('data/Detection/test_images/*.jpg'):
        test_images.append(cv2.imread(filename, cv2.IMREAD_COLOR))

    if test_images:
        if len(test_images) > 4:  # only 4 images allowed at a time
            test_images = test_images[:4]

        m = len(test_images)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        _, axes = plt.subplots(2, m, figsize=(m * 4, 8))
        axes = axes.flatten()
        for i, image in enumerate(test_images):
            visual_test_on_single_image(image, detr_model, device, axes[i], 'DETR')
            visual_test_on_single_image(image, frcnn_model, device, axes[i + m], 'FRCNN')
        plt.show()
    else:
        print('[!] No test images in "data/Detection/test_images/" with format .jpg')
