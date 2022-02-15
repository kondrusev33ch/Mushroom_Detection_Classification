"""
Resources:
    https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train/notebook
    https://www.kaggle.com/ChristianDenich/end2end-object-detection-with-fasterrcnn
"""

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection_helpers import AverageMeter, calculate_image_precision
from tqdm import tqdm

LR = 0.00002


# Model
# =====================================================================================================
def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Training Function
# =====================================================================================================
def train_fn(data_loader, model, optimizer, device, scheduler, batch_size):
    model.train()
    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss.to(torch.float32) for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), batch_size)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


# Eval function
# =====================================================================================================
def eval_fn(data_loader, model, device):
    model.eval()
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    image_precisions = []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # In eval mode model doesn't return losses, so we need another way to validate model
            outputs = model(images)
            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = boxes[preds_sorted_idx]
                image_precision = calculate_image_precision(preds_sorted,
                                                            gt_boxes,
                                                            thresholds=iou_thresholds,
                                                            form='pascal_voc')
                image_precisions.append(image_precision)
    valid_precision = np.mean(image_precisions)
    return valid_precision


# Main
# =====================================================================================================
def run(train_data_loader, valid_data_loader, epochs, batch_size):
    frcnn_model = get_model()
    device = torch.device('cuda')
    model = frcnn_model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    t_losses, v_precisions = [], []

    best_precision = 0.01
    for epoch in range(epochs):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler=None,
                              batch_size=batch_size)
        valid_precision = eval_fn(valid_data_loader, model, device)
        print('| EPOCH {} | TRAIN_LOSS {} | VALID_PRECISION {} |'.format(
            epoch, train_loss.avg, valid_precision))

        t_losses.append(train_loss.avg)
        v_precisions.append(valid_precision)

        if valid_precision > best_precision:
            best_precision = valid_precision
            print('Best model found in Epoch {} ......... Saving Model'.format(epoch))
            torch.save(model.state_dict(), f'frcnn_best_{epoch}.pth')

    return model, t_losses, v_precisions


if __name__ == '__main__':
    pass
