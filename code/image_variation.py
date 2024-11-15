import cv2
import numpy as np
import random

# Rotación aleatoria
def random_rotation(image):
    angle = random.uniform(-10, 10)  
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))


# Aumento de brillo y contraste aleatorios
def random_brightness_contrast(image):
    # Contraste aleatorio
    alpha = 1.0 + random.uniform(-0.2, 0.2)
    # Brillo aleatorio
    beta = random.uniform(-50, 50)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted
    
# Recorte y ajuste de tamaño aleatorio
def random_resized_crop(image):
    h, w = image.shape[:2]
    scale = random.uniform(0.8, 1.0)  # Escala aleatoria entre 0.8 y 1.0
    new_h, new_w = int(h * scale), int(w * scale)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped = image[top:top + new_h, left:left + new_w]
    resized = cv2.resize(cropped, (416, 416))
    return resized

# Añadir ruido gaussiano
def add_gaussian_noise(image):
    mean = 0
    std_dev = 0.4 #random.uniform(0.5)
    #r, c, ch = image.shape
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def create_image_variations(img):
    #img = cv2.resize(img, (256, 256))
    img = random_rotation(img)
    img = random_brightness_contrast(img)
    img = random_resized_crop(img)
    if random.random() > 0.7:
        img = add_gaussian_noise(img)
    return img


