from torchvision import models
import torch.nn as nn
import torch


def createModel():
    """Create model optimized for grayscale emotion recognition"""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)

    # Modify first conv layer for single-channel grayscale input
    original_conv1 = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )

    # Initialize with averaged pretrained weights
    with torch.no_grad():
        model.features[0][0].weight = nn.Parameter(
            original_conv1.weight.mean(dim=1, keepdim=True)
        )

    model.classifier[1] = nn.Linear(in_features=1280, out_features=7, bias=True)

    for param in model.parameters():
        param.requires_grad = False

    return model
