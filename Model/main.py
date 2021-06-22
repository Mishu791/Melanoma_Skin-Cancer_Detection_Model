import os
import torch

import albumentations
import pretrainedmodels

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torch.nn import functional as F
                                                                                #from apex import amp
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping                                           #import necessary dependencies

##for installing packages and libraries use python -m pip install -U packagename
## before that install pip via python -m pip install -U pip
## https://discuss.atom.io/t/how-to-install-python-modules/47622/2



#we are gonna use pretrained model

class SEResNext50_32x4d(nn.Module):                                             #defining class
    def __init__(self, pretrained= 'imagenet'):                                 #if pretained = imagenet then load the pretrained weights
        super(SEResNext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "D:\Atom_project\Melanoma\se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):                                                              #forward function will take batch of images
        bs, _, _, _ = image.shape                                                                   #calculating the batch size, channel, height and width by image.shape
        x = self.model.features(image)                                                              #getting the features of batch of the images
        x = F.adaptive_avg_pool2d(x, 1)                                                             #selecting adaptive_avg_pool2d otherwise model size would not be same
        x = x.reshape(bs, -1)                                                                       #reshaping to the batch size
        out = self.out(x)                                                                           # creating output
        loss = nn.BCEWithLogitsloss()(
            out, targets.reshape(-1, 1).type_as(out)
        )
        return out

def train(fold):
    training_data_path = "D:/Atom_project/Melanoma/data_melanoma/train_224/x_train_224_mein.npy"
    model_path = "D:/Atom_project/Melanoma"
    df = pd.read_csv("D:/Atom_project/Melanoma/data_melanoma/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold!= fold].reset_index(drop= True)
    df_valid = df[df.kfold== fold].reset_index(drop= True)

    train_aug = albumentations.Compose(                                                              # using albumentations for normalizing the image
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )


    train_images = df_train.image_name.values.tolist()                                               # making image_name into a list
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values


    train_dataset = ClassificationLoader(
        image_paths = train_images,
        targets = train_targets,
        resize = None,
        augmentations = train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=False,
        num_workers=4
        )


    valid_dataset = ClassificationLoader(
        image_paths = valid_images,
        targets = valid_targets,
        resize = None,
        augmentations = valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained = 'imagenet')                          # model name
    model.to(device)                                                            # give an argument 'device'


    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)                 # using optimizer: Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        verbose = True,
        mode="max"
    )

    es = EarlyStopping(patience=5, verbose = False, mode='max')
    for epoch in range (epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device,
            fp16=True
            )

        predictions, valid_loss = Engine.evaluate(
            train_loader,
            model,
            optimizer,
            device
            )

        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch = {epoch}, auc = {auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break

if __name__ == "__main__":
    train(fold = 0)
