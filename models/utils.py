import numpy as np
import os
from PIL import Image, ImageOps

def load_image(image_path, expand_dims=False):
    image = Image.open(image_path)
    image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image.astype('float32')
    return image

def image_generator(train_data_path, batch_size):
    """
    Creates a generator for the images in the train data folder.
    """
    filenames = os.listdir(train_data_path)
    np.random.shuffle(filenames)
    filename_gen = (f for f in filenames)
    while True:
        batch = []
        while len(batch) < batch_size:
            image_path = next(filename_gen)
            image = load_image(os.path.join(train_data_path, image_path))
            if image.shape[-1] == 3:
                batch.append(image)
        yield np.array(batch)

if __name__ == "__main__":
    puppy_image = load_image('../images/content/puppy.jpg')
    gen = image_generator('../data/train2014', 4)
    for x in gen:
        if x.shape != (4, 256, 256, 3):
            print("Bad image")

