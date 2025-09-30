import torch
import timm
from timm.layers import Conv2dSame, SelectAdaptivePool2d
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time


def createModel():
    """Create model optimized for grayscale emotion recognition"""
    model = models.mobilenet_v2(weights=None)

    # Modify first conv layer for single-channel grayscale input
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )

    model.classifier[1] = nn.Linear(in_features=1280, out_features=7, bias=True)

    for param in model.parameters():
        param.requires_grad = False

    return model


def createLite():
    model = timm.create_model("efficientnet_lite0", pretrained=False)

    # Modify first conv layer for single-channel grayscale inputs
    old_conv = model.conv_stem
    model.conv_stem = Conv2dSame(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        bias=(old_conv.bias is not None),
    )

    model.classifier = nn.Linear(in_features=1280, out_features=7, bias=True)

    for param in model.parameters():
        param.requires_grad = False

    return model


def saveScript(img_path):
    # model = createModel()
    model = createLite()

    weights_path = "/home/fw7th/emotion/data/efficientlite/lite.pth"
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    emotion_dict = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "suprise",
    }

    img = Image.open(img_path).convert("L")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                (112, 112), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    img_tensor = transform(img)
    input = img_tensor.unsqueeze(0)
    print(f"Shape: {input.size()}")

    model.to("cpu")
    input.to("cpu")

    start_time = time.time()
    with torch.no_grad():
        output = model(input)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    end_time = time.time()
    inference_time = end_time - start_time

    print(probs)
    print(f"Prediction: {emotion_dict[pred]}\n")
    print(f"Time taken for inference: {inference_time:.4f} seconds")

    dummy_input = torch.randn(1, 1, 112, 112)
    traced_model = torch.jit.trace(model, dummy_input)

    # Save TorchScript model
    traced_model.save("/home/fw7th/emotion/data/efficientlite/lite.pt")


saveScript("/home/fw7th/isolation/data/disgust1.jpg")
