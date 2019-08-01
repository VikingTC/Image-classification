import sys
import tarfile
from six.moves import urllib
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import matplotlib.image as image
import numpy as np
FLOWER_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
FLOWER_PATH = os.path.join('datasets', 'flowers')

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_flowers(url=FLOWER_URL, path=FLOWER_PATH):
    if os.path.exists(FLOWER_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)

fetch_flowers()

flowers_root_path = os.path.join(FLOWER_PATH, 'flower_photos')
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])
print(flower_classes)
# get list all images path for each class
image_paths = defaultdict(list)
for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith('.jpg'):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))
for paths in image_paths.values():
    paths.sort()
# plot a few image for each class
n_examples = 2

for flower_class in flower_classes:
    print("Class:", flower_class)
    plt.figure(figsize=(10, 5))
    for index, example_image_path in enumerate(image_paths[flower_class][:n_examples]):
        example_image = image.imread(example_image_path)
        plt.subplot(100 + n_examples * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()
