import numpy as np
from PIL import Image
import random

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

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
image = np.power(np.clip(image, 1e-8, 1), gamma)
            image = np.clip(image, 0, 1)
    
    return image

def shadow_direction_augmentation(image):
    h, w = image.shape[:2]
    
    if random.random() < 0.4:
        shadow_type = random.choice(['hard', 'soft', 'directional', 'partial'])
        
        if shadow_type == 'hard':
            y_pos = random.randint(0, h)
            darkness = random.uniform(0.3, 0.7)
            mask = np.linspace(np.ones(w), darkness * np.ones(w), h)
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            
        elif shadow_type == 'soft':
            y_pos = random.randint(0, h)
            x_pos = random.randint(0, w)
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((yy - y_pos)**2 + (xx - x_pos)**2).astype(float)
            max_dist = np.sqrt(h**2 + w**2)
            darkness = random.uniform(0.4, 0.8)
            mask = np.clip(dist / max_dist * darkness + (1 - darkness), 0, 1)
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            
        elif shadow_type == 'directional':
            angle = random.uniform(0, np.pi)
            x_shift = (np.arange(h)[:, np.newaxis] * np.cos(angle)).astype(float)
            y_shift = (np.arange(w)[np.newaxis, :] * np.sin(angle)).astype(float)
            x_grad = (x_shift / h * random.uniform(0.3, 0.7))
            y_grad = (y_shift / w * random.uniform(0.3, 0.7))
            mask = 1 - (x_grad + y_grad)
            mask = np.clip(np.tile(mask[:, :, np.newaxis], (1, 1, 3)), 0.1, 1)
            
        else:  # partial
            center_x = random.randint(w // 4, 3 * w // 4)
            center_y = random.randint(h // 4, 3 * h // 4)
            radius = random.randint(min(h, w) // 4, min(h, w) // 2)
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)
            mask = np.ones((h, w, 3))
            dark_mask = dist < radius
            darkness = random.uniform(0.3, 0.6)
            mask[dark_mask] = darkness
        
        if 'mask' in dir():
            if len(mask.shape) == 2:
                mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            image = np.clip(image * mask, 0, 1)
    
    if random.random() < 0.3:
        light_from_left = random.uniform(0.7, 1.0)
        light_from_right = random.uniform(0.7, 1.0)
        gradient_x = np.linspace(light_from_left, light_from_right, w)
        gradient = np.tile(gradient_x[np.newaxis, :, np.newaxis], (h, 1, 3))
        image = np.clip(image * gradient, 0, 1)
    
    if random.random() < 0.3:
        light_from_top = random.uniform(0.7, 1.0)
        light_from_bottom = random.uniform(0.7, 1.0)
        gradient_y = np.linspace(light_from_top, light_from_bottom, h)
        gradient = np.tile(gradient_y[:, :, np.newaxis], (1, w, 3))
        image = np.clip(image * gradient, 0, 1)
    
    return image

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
