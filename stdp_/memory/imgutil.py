import numpy as np
from skimage.draw import rectangle
from PIL import Image
from matplotlib import pyplot as plt
import os
import math
import random
from skimage.metrics import structural_similarity as ssim

def image_to_array(path):
    img = np.array(Image.open(path).resize((64, 64)))
    img, alpha = (np.ones((64, 64, 3), dtype=np.int8) * 255 - img[:, :, :3]).astype(np.float32) / 255, img[:, :, 3].astype(np.float32)
    img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
    img = img * alpha
    return img

def images_to_arrays(folder_path):
    li = []
    names = []
    for cur, _, files in os.walk(folder_path):
        for file in files:
            if file[-4:] == '.png':
                li.append(image_to_array(os.path.join(cur, file)))
                names.append(file[:-4])
    return li, names

def encode(arrays):
    if isinstance(arrays, np.ndarray):
        return np.reshape(arrays, (64 * 64))
    return np.reshape(np.stack(arrays, 2), (64 * 64 * len(arrays)))

def decode(array: np.array):
    array = np.reshape(array, (64, 64, array.shape[0] // (64 * 64)))
    return [array[:, :, i] for i in range(array.shape[2])]

def show_gray_images(arrays, name=None):
    num = math.ceil(len(arrays) ** 0.5)
    for i in range(len(arrays)):
        plt.subplot(num, num, i+1)
        if name is not None:
            plt.title(name[i])
        plt.imshow(arrays[i], cmap='gray')
    plt.show()

def draw_box(arrays):
    if isinstance(arrays, np.ndarray):
        mean = np.mean(arrays)
        s1, s2, e1, e2 = 16, 16, 47, 47
        rr, cc = rectangle((s1, s2), (e1, e2))
        arrays[rr, cc] = mean
        return arrays
    return list(map(draw_box, arrays))

alpha = 0.25
def add_noise(arrays):
    if isinstance(arrays, np.ndarray):
        arrays /= 255
        noise = np.random.normal(size=(64, 64))
        res = arrays * (1 - alpha ** 2) ** 0.5 + alpha * noise * np.linalg.norm(arrays) / (1 << 6)
        return np.minimum(np.maximum(res, np.zeros((64, 64))), np.ones((64, 64))) * 255
    return list(map(add_noise, arrays))

def accuracy(img1, img2):
    if isinstance(img1, np.ndarray):
        img1 = img1 - np.ones(img1.shape) * np.mean(img1)
        img2 = img2 - np.ones(img2.shape) * np.mean(img2)
        img1 = img1 / np.linalg.norm(img1)
        img2 = img2 / np.linalg.norm(img2)
        return (np.sum(img1 * img2) + 1) / 2
    return list(map(lambda x: accuracy(*x), zip(img1, img2)))

def another_accuracy(img1, img2):
    if isinstance(img1, np.ndarray):
        img1 = img1 / np.linalg.norm(img1)
        img2 = img2 / np.linalg.norm(img2)
        return np.sum(img1 * img2)
    return list(map(lambda x: another_accuracy(*x), zip(img1, img2)))

def accuracy_ssim(img_src, img_mod):
    if isinstance(img_src, np.ndarray):
        return ssim(img_src, img_mod, data_range=img_mod.max() - img_mod.min())
    return list(map(lambda x: another_accuracy(*x), zip(img_src, img_mod)))

def show_gray_images_horizontal(arrays, name=None):
    for i in range(len(arrays)):
        plt.subplot(1, len(arrays), i+1)
        if name is not None:
            plt.title(name[i])
        plt.imshow(arrays[i], cmap='gray')
    plt.show()
if __name__ == '__main__':
    show_gray_images(decode(encode(add_noise(images_to_arrays('images')[0]))))
