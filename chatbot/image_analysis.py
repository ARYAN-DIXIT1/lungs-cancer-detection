from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Load BLIP Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to Generate Image Caption
def get_image_description(image):
    image = image.convert("RGB")  # Ensure RGB format
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return description
