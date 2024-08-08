import torch
from PIL import Image
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

def initialize_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    return device, model, feature_extractor

# Initialize the model, feature extractor, and device globally
device, model, feature_extractor = initialize_model("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

def process_and_predict(image_path, model, feature_extractor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=image.size[::-1], 
            mode='bilinear', 
            align_corners=False
        )

    predicted_class = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    utility_pole_class_index = 5
    binary_mask = np.where(predicted_class == utility_pole_class_index, 1, 0)
    return binary_mask