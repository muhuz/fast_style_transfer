import numpy as np
import os
from PIL import Image, ImageOps

def load_image(image_path):
    image = Image.open(image_path)
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    puppy_image = load_image('../images/content/puppy.jpg')
