import torch
from torchvision import transforms
from PIL import Image
import time
from .model import createModel


def saveScript(img_path):
    model = createModel()

    weights_path = "/home/fw7th/emotions/data/mobilenet-v3/mobilenet_v3.pth"
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
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.BICUBIC
            ),
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

    dummy_input = torch.randn(1, 1, 64, 64)
    traced_model = torch.jit.trace(model, dummy_input)

    # Save TorchScript model
    traced_model.save("/home/fw7th/emotions/data/mobilenet-v3/v3.pt")


saveScript("/home/fw7th/isolation/data/disgust.jpeg")
