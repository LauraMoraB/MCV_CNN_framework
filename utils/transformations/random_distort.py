import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class Random_distort(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, brightness_delta=32/255., contrast_delta=0.5, saturation_delta=0.5, hue_delta=0.1):
        '''A color related data augmentation used in SSD.

        Args:
          img: (PIL.Image) image to be color augmented.
          brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
          contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
          saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
          hue_delta: (float) shift of hue, range from [-delta,delta].

        Returns:
          img: (PIL.Image) color augmented image.
        '''
        def brightness(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(brightness=delta)(img)
            return img

        def contrast(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(contrast=delta)(img)
            return img

        def saturation(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(saturation=delta)(img)
            return img

        def hue(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(hue=delta)(img)
            return img

        if self.cf.random_dist:
            if not isinstance(img, Image.Image):
                if img.dtype != 'uint8':
                    img = img.astype('uint8')
                img = Image.fromarray(img)
            img = brightness(img, brightness_delta)
            if random.random() < 0.5:
                img = contrast(img, contrast_delta)
                img = saturation(img, saturation_delta)
                img = hue(img, hue_delta)
            else:
                img = saturation(img, saturation_delta)
                img = hue(img, hue_delta)
                img = contrast(img, contrast_delta)
        return np.array(img)
