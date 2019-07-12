# fast_style_transfer

My implementation of Fast Style Transfer described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

The model uses VGG19 to extract the style and content layers. The network is trained using the COCO 2014 Dataset. Here is a gif of the model's evaluation on an image of a puppy as it gets trained over time.

![Alt Text](images/gifs/puppy_wave.gif)
