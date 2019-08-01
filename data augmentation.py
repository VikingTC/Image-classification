import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import defaultdict

from PIL import Image
import matplotlib.image as image
def rotate(img, angle=90, scale=1.0):
    w = img.shape[1]
    h =img.shape[0]
    # rotate matrix
    M = cv2.M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    image = cv2.warpAffine(img, M, (w, h))
    return image

def flip(image, vflip=False, hflip=False):
    if hflip or vflip:
        if hflip and vflip:
            c = -1
        else:
            c = 0 if vflip else 1
        image = cv2.flip(image, flipCode=c)
    return image


def add_gaussian_noise(img):
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy

def image_augment(img, name, name_int, save_path):
    img_flip = flip(img, vflip=True, hflip=False)
    img_rot = rotate(img)
    img_gaussian = add_gaussian_noise(img)

    width = 299
    height = 299
    dim = (width, height)
    # resize image
    img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_flip_resize = cv2.resize(img_flip, dim, interpolation=cv2.INTER_AREA)
    img_rot_resize = cv2.resize(img_rot, dim, interpolation=cv2.INTER_AREA)
    img_gaussian_resize = cv2.resize(img_gaussian, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path + '/%s' % str(name) + '/%s' % str(name) + '.jpg', img_resize)
    cv2.imwrite(save_path + '/%s' % str(name) + '/%s' % str(name_int) + '_vflip.jpg', img_flip_resize)
    cv2.imwrite(save_path + '/%s' % str(name) + '/%s' % str(name_int) + '_rot.jpg', img_rot_resize)
    cv2.imwrite(save_path + '/%s' % str(name) + '/%s' % str(name_int) + '_GaussianNoise.jpg', img_gaussian_resize)

FLOWER_PATH = os.path.join('datasets', 'flowers')

flowers_root_path = os.path.join(FLOWER_PATH, 'flower_photos')
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])

image_paths = defaultdict(list)
for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith('.jpg'):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

for paths in image_paths.values():
    paths.sort()
n_samples = 100
FLOWER_PATH_AUG = os.path.join('datasets/flowers aug')
if os.path.exists(FLOWER_PATH_AUG) is False:
    os.makedirs(FLOWER_PATH_AUG, exist_ok=True)
for flower_class in flower_classes:
    FLOWER_PATH_AUG_ROOT = os.path.join('datasets/flowers aug', flower_class)
    if os.path.exists(FLOWER_PATH_AUG_ROOT) is False:
        os.makedirs(FLOWER_PATH_AUG_ROOT, exist_ok=True)

    for index, image_path in enumerate(image_paths[flower_class]):
        img = image.imread(image_path)

        image_augment(img, flower_class, (flower_class+str(index)), FLOWER_PATH_AUG)
