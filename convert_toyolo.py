import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from sklearn.model_selection import train_test_split

ROOT_DIR = os.getcwd()


def get_all_class(file_list, label_path=ROOT_DIR):
    classes = list()
    for filename in file_list:
        json_path = os.path.join(label_path, filename + '.json')
        json_file = json.load(open(json_path, "r", encoding="utf-8"))
        for item in json_file["shapes"]:
            label_class = item['label']
            if label_class not in classes:
                classes.append(label_class)
    return classes


def split_dataset(label_path, isUseTest=False):
    files = glob(label_path + "\\*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    test_files = None
    if isUseTest:
        trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
    else:
        trainval_files = files
    train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)
    return train_files, val_files, test_files, files


def create_save_file(label_path=ROOT_DIR):
    # 生成训练集
    train_image = os.path.join(label_path, 'train', 'images')
    if not os.path.exists(train_image):
        os.makedirs(train_image)
    train_label = os.path.join(label_path, 'train', 'labels')
    if not os.path.exists(train_label):
        os.makedirs(train_label)
    # 生成验证集
    val_image = os.path.join(label_path, 'valid', 'images')
    if not os.path.exists(val_image):
        os.makedirs(val_image)
    val_label = os.path.join(label_path, 'valid', 'labels')
    if not os.path.exists(val_label):
        os.makedirs(val_label)
    # 生成测试集
    test_image = os.path.join(label_path, 'test', 'images')
    if not os.path.exists(test_image):
        os.makedirs(test_image)
    test_label = os.path.join(label_path, 'test', 'labels')
    if not os.path.exists(test_label):
        os.makedirs(test_label)
    return train_image, train_label, val_image, val_label, test_image, test_label


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def push_into_file(file, images, labels, label_path=ROOT_DIR, suffix='.png'):
    for filename in file:
        image_file = os.path.join(label_path, filename + suffix)
        label_file = os.path.join(label_path, filename + '.txt')
        if not os.path.exists(os.path.join(images, filename + suffix)):
            try:
                shutil.move(image_file, images)
            except OSError:
                pass
        if not os.path.exists(os.path.join(labels, filename + suffix)):
            try:
                shutil.move(label_file, labels)
            except OSError:
                pass


def json2txt(classes, txt_Name='allfiles', label_path=ROOT_DIR, suffix='.png'):
    store_json = os.path.join(label_path, 'json')
    if not os.path.exists(store_json):
        os.makedirs(store_json)

    _, _, _, files = split_dataset(label_path)
    if not os.path.exists(os.path.join(label_path, 'tmp')):
        os.makedirs(os.path.join(label_path, 'tmp'))

    list_file = open('tmp/%s.txt' % txt_Name, 'w')
    for json_file_ in files:
        json_filename = os.path.join(label_path, json_file_ + ".json")
        imagePath = os.path.join(label_path, json_file_ + suffix)
        list_file.write('%s\n' % imagePath)
        out_file = open('%s/%s.txt' % (label_path, json_file_), 'w')
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        if os.path.exists(imagePath):
            height, width, channels = cv2.imread(imagePath).shape
            for multi in json_file["shapes"]:
                points = np.array(multi["points"])
                xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
                xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
                ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
                ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
                label = multi["label"]
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    cls_id = classes.index(label)
                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = convert((width, height), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                    # print(json_filename, xmin, ymin, xmax, ymax, cls_id)
        if not os.path.exists(os.path.join(store_json, json_file_ + '.json')):
            try:
                shutil.move(json_filename, store_json)
            except OSError:
                pass


def create_yaml(classes,label_path,isUseTest=False):
    nc = len(classes)
    if not isUseTest:
        desired_caps = {
            'path':label_path,
            'train': 'train/images',
            'val': 'valid/images',
            'nc': nc,
            'names': classes
        }
    else:
        desired_caps = {
            'path':label_path,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': nc,
            'names': classes
        }
    yamlpath = os.path.join(label_path, "data" + ".yaml")

    with open(yamlpath, "w+", encoding="utf-8") as f:
        for key,val in desired_caps.items():
            yaml.dump({key:val}, f, default_flow_style=False)


def ChangeToYolo5(label_path=ROOT_DIR, suffix='.png', isUseTest=False):
    train_files, val_files, test_file, files = split_dataset(label_path, isUseTest=isUseTest)
    classes = get_all_class(files)
    json2txt(classes)
    create_yaml(classes, label_path, isUseTest=isUseTest)
    train_image, train_label, val_image, val_label, test_image, test_label = create_save_file(label_path)
    push_into_file(train_files, train_image, train_label, suffix=suffix)
    push_into_file(val_files, val_image, val_label, suffix=suffix)
    if test_file is not None:
        push_into_file(test_file, test_image, test_label, suffix=suffix)
    print('create dataset done')


if __name__ == "__main__":
    ChangeToYolo5()