import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def save_segmentation(binary_mask, output_path):
    plt.imsave(output_path, binary_mask, cmap='gray')

def overlay_segmentation_on_image(image, binary_mask):
    overlay = Image.fromarray((binary_mask * 255).astype(np.uint8)).convert("RGBA")
    image = image.convert("RGBA")
    return Image.blend(image, overlay, alpha=0.5)

def save_overlay_image(image, binary_mask, output_path):
    overlayed_image = overlay_segmentation_on_image(image, binary_mask)
    overlayed_image.save(output_path)