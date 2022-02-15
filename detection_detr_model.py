"""
Resources:
    https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr#About-this-Notebook

Requirements:
    !git clone https://github.com/facebookresearch/detr.git
    for loss function
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from detection_helpers import AverageMeter
import sys

sys.path.append('../detr/')
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

NUM_CLASSES = 2
NUM_QUERIES = 30
NUM_CLASS_COEF = 0.5
LR = 0.00002


# Model
# =====================================================================================================
class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


def get_model():
    return DETRModel(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES)


# Training Function
# =====================================================================================================
def train_fn(data_loader, model, criterion, optimizer, device, scheduler, batch_size):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
def eval_fn(data_loader, model, criterion, device, batch_size):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)
            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            summary_loss.update(losses.item(), batch_size)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


# Main
# =====================================================================================================
def run(train_data_loader, valid_data_loader, epochs, batch_size):
    detr_model = get_model()
    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']

    device = torch.device('cuda')
    model = detr_model.to(device)
    criterion = SetCriterion(NUM_CLASSES - 1, matcher, weight_dict,
                             eos_coef=NUM_CLASS_COEF, losses=losses)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    t_losses, v_losses = [], []

    best_loss = 10 ** 5
    for epoch in range(epochs):
        train_loss = train_fn(train_data_loader, model, criterion, optimizer, device,
                              scheduler=None, batch_size=batch_size)
        valid_loss = eval_fn(valid_data_loader, model, criterion, device, batch_size)
        print('| EPOCH {} | TRAIN_LOSS {} | VALID_LOSS {} |'.format(
            epoch, train_loss.avg, valid_loss.avg))

        t_losses.append(train_loss.avg)
        v_losses.append(valid_loss.avg)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found in Epoch {} ......... Saving Model'.format(epoch))
            torch.save(model.state_dict(), f'detr_best_{epoch}.pth')

    return model, t_losses, v_losses


if __name__ == '__main__':
    pass
