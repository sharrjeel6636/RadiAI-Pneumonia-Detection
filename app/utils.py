import base64
import io
import numpy as np
import pydicom
from PIL import Image

def base64_to_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Invalid image format: {e}")

def dicom_to_image(dicom_bytes: bytes) -> Image.Image:
    try:
        dataset = pydicom.dcmread(io.BytesIO(dicom_bytes))
        image_data = dataset.pixel_array.astype(float)
        # Normalize to 0-255
        image_data = (np.maximum(image_data, 0) / image_data.max()) * 255.0
        return Image.fromarray(image_data.astype(np.uint8)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid DICOM file: {e}")

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
