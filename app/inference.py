import torch
from torchvision import models, transforms
from captum.attr import LayerGradCam
from .utils import base64_to_image, dicom_to_image, image_to_base64
from PIL import Image
import numpy as np
import io

# For simplicity, using a pre-trained ResNet18
# TODO: Replace with your actual model checkpoint loading logic
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_inference(file_bytes: bytes, is_dicom: bool = False):
    try:
        if is_dicom:
            image = dicom_to_image(file_bytes).convert('RGB')
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

        input_tensor = transform(image).unsqueeze(0)

        # Inference
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

        # Grad-CAM
        # Target the last convolutional layer
        target_layer = model.layer4[1].conv2
        grad_cam = LayerGradCam(model, target_layer)
        attr = grad_cam.attribute(input_tensor, target=prediction)

        # Process heatmap
        attr = attr.detach().squeeze().numpy()
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        heatmap = Image.fromarray((attr * 255).astype(np.uint8)).resize(image.size)

        return str(prediction), image_to_base64(heatmap)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
