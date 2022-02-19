import os
import numpy as np
import json
import shutil
import cv2
from tqdm import tqdm

ROOT_PATH = 'datasets/mix_det/'
DATA_PATH = os.path.join(ROOT_PATH, 'annotations/train.json')
print(DATA_PATH)
IMGS_PATH = os.path.join(ROOT_PATH, 'images/train/')
LABELS_PATH = os.path.join(ROOT_PATH, 'labels/train/')
os.makedirs(IMGS_PATH, exist_ok=True)
os.makedirs(LABELS_PATH, exist_ok=True)

data = json.load(open(DATA_PATH))
ext = 'jpg'
imgs_size = {}
for i in tqdm(range(len(data['images']))):
    img_id = data['images'][i]['id']
    img_name = data['images'][i]['file_name']
    ext = img_name.split('.')[-1]
    img_path = os.path.join(ROOT_PATH, img_name)
    img = cv2.imread(img_path)
    img_size = img.shape[:2]
    imgs_size[img_id] = img_size
    # print(img_path)
    new_path = IMGS_PATH + str(img_id) + '.' + ext
    shutil.move(img_path, new_path)
    labels = LABELS_PATH + str(img_id) + '.txt'
    # open(labels, 'w').close()

for i in tqdm(range(len(data['annotations']))):
    img_id = data['annotations'][i]['image_id']
    img_size = imgs_size[img_id]
    x = data['annotations'][i]['bbox'][0]
    y = data['annotations'][i]['bbox'][1]
    w = data['annotations'][i]['bbox'][2]
    h = data['annotations'][i]['bbox'][3]
    if (x < 0):
        w = w + x
        x = 0
    if (y < 0):
        h = h + y
        y = 0
    if (x + w > img_size[1]):
        w = img_size[1] - x
    if (y + h > img_size[0]):
        h = img_size[0] - y
    x = x + w/2
    y = y + h/2
    x = x/img_size[1]
    y = y/img_size[0]
    w = w/img_size[1]
    h = h/img_size[0]
    label = data['annotations'][i]['category_id']
    label = str(label - 1)
    label_path = os.path.join(LABELS_PATH, str(img_id) + '.txt')
    with open(label_path, 'a') as f:
        f.write(label + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
