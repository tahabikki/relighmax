import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image
import random
import warnings
warnings.filterwarnings('ignore')

def data_augmentation(image, mode):
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))
    return image

def random_augmentation(image):
    if random.random() < 0.5:
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            image = np.clip(image * brightness_factor, 0, 1)
        else:
            contrast_factor = random.uniform(0.7, 1.3)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 1)
    if random.random() < 0.3:
        hue_shift = random.uniform(-0.1, 0.1)
        image = np.clip(image + hue_shift, 0, 1)
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.3)
        image = np.clip(np.power(image + 1e-8, gamma), 0, 1)
    return image

def shadow_direction_augmentation(image):
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) > 2 else 1
    
    if random.random() < 0.4:
        shadow_type = random.choice(['hard', 'soft', 'directional', 'partial'])
        if shadow_type == 'hard':
            darkness = random.uniform(0.3, 0.7)
            mask = np.linspace(1.0, darkness, h)
            mask = np.outer(mask, np.ones(w))
            if c == 3:
                mask = np.stack([mask] * 3, axis=2)
            else:
                mask = mask[:, :, np.newaxis]
            image = np.clip(image * mask, 0, 1)
        elif shadow_type == 'soft':
            y_pos = random.randint(0, h)
            x_pos = random.randint(0, w)
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((yy - y_pos)**2 + (xx - x_pos)**2).astype(float)
            max_dist = np.sqrt(h**2 + w**2)
            darkness = random.uniform(0.4, 0.8)
            mask = np.clip(dist / max_dist * darkness + (1 - darkness), 0, 1)
            if c == 3:
                mask = np.stack([mask] * 3, axis=2)
            else:
                mask = mask[:, :, np.newaxis]
            image = np.clip(image * mask, 0, 1)
        elif shadow_type == 'directional':
            angle = random.uniform(0, np.pi)
            x_grad = np.cos(angle) * np.linspace(0.3, 0.7, h)[:, np.newaxis]
            y_grad = np.sin(angle) * np.linspace(0.3, 0.7, w)[np.newaxis, :]
            mask = 1 - (x_grad + y_grad)
            if c == 3:
                mask = np.stack([mask] * 3, axis=2)
            else:
                mask = mask[:, :, np.newaxis]
            mask = np.clip(mask, 0.1, 1)
            image = np.clip(image * mask, 0, 1)
    if random.random() < 0.3:
        light_left = random.uniform(0.7, 1.0)
        light_right = random.uniform(0.7, 1.0)
        gradient = np.linspace(light_left, light_right, w)
        gradient = gradient[np.newaxis, :] * np.ones((h, 1))
        if c == 3:
            gradient = np.stack([gradient] * 3, axis=2)
        image = np.clip(image * gradient, 0, 1)
    if random.random() < 0.3:
        light_top = random.uniform(0.7, 1.0)
        light_bottom = random.uniform(0.7, 1.0)
        gradient = np.linspace(light_top, light_bottom, h)
        gradient = gradient[:, np.newaxis] * np.ones((1, w))
        if c == 3:
            gradient = np.stack([gradient] * 3, axis=2)
        image = np.clip(image * gradient, 0, 1)
    return image

def load_images(file_path):
    im = Image.open(file_path)
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    elif im.mode == 'L':
        im = im.convert('RGB')
    arr = np.array(im, dtype="float32") / 255.0
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=2)
    return arr

def save_images(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    if result_2 is None or not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)
    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')