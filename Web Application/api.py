# way to upload a image : endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch

from flask import Flask
from flask import request
from flask import render_template

import albumentations
import pretrainedmodels

import numpy as np
import torch.nn as nn

from torch.nn import functional as F

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine


app = Flask(__name__)
upload_folder = "D:\Atom_project\Web_App_melanoma\static"
# directory where the file will be saved
DEVICE = "cuda"
Model = None


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch_size, -1)
        out = torch.sigmoid(self.l0(x))
        loss = 0

        return out, loss


def predict(image_path, Model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        #as we have only one image to predict
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    predictions = Engine.predict(test_loader, model, DEVICE)
    predictions = np.vstack((predictions)).ravel()

    return predictions



@app.route("/", methods=["GET", "POST"])
# app will be running on GET and POST request
def upload_predict():
    # it will only request image from file we provided in HTML name 'image' if the request method is of kind 'POST'
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            #also we are checking whether the image file does exist or not
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, Model)[0]
            # print(pred)
            # if image file exists then we are saving the image in a 'image_location'
            return render_template("index.html", prediction = pred, image_loc= image_file.filename)
            # if exists then render will give a prediction value of 1
    return render_template("index.html", prediction = 0, image_loc=None)

if __name__ == "__main__":
        Model = SEResnext50_32x4d(pretrained=None)
        Model.load_state_dict(torch.load("D:/Atom_project/Web_App_melanoma/model_fold_0.bin"))
        #https://pytorch.org/tutorials/beginner/saving_loading_models.html
        Model.to(DEVICE)
        app.run(port=12000, debug = True)
