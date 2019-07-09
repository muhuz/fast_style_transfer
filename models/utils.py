import numpy as np
import os
from PIL import Image, ImageOps
import scipy.misc

def load_image(image_path, expand_dims=False, fit=True):
    image = Image.open(image_path)
    if fit:
        image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image.astype('float32')
    return image

def save_image(image_path, image):
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(image_path, image)

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
        yield np.array(batch, dtype=np.float32)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    puppy_image = load_image('../images/content/puppy.jpg')
    gen = image_generator('../data/train2014', 4)
    k = 0
    for x in gen:
        fig, axes = plt.subplots(2, 2)
        for i in range(4):
            j, k = i % 2, i // 2
            axes[j,k].imshow(x[i].astype(int))
        plt.show()
        k += 1
        if k == 10:
            break


