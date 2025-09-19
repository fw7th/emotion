import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import time


def FER_image(img_path):
    num_classes = 7

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Sequential(
        nn.BatchNorm1d(model.fc.in_features),
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 7),
    )

    weights_path = "/home/fw7th/emotions/data/emotion_resnet/heavymodel_best.pth"
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
            transforms.Resize((112, 112)),
            transforms.Grayscale(num_output_channels=1),
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

    dummy_input = torch.randn(1, 1, 112, 112)  # depends on training resolution
    traced_model = torch.jit.trace(model, dummy_input)

    # Save TorchScript model
    ##traced_model.save("/home/fw7th/emotions/data/emotion_resnet/emotion.pt")


FER_image("/home/fw7th/Pictures/me.jpg")
