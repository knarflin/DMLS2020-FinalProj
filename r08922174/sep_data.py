#!/usr/bin/env python3

import os
import shutil
import pdb
import random
import numpy as np

from pathlib import Path


random.seed(0)
np.random.seed(0)


def main():

    original_path = "./data/train"
    a = os.walk(original_path)

    image_files = []

    for dirname, dirnames, filenames in a:
        for filename in filenames:
            image_file = os.path.join(dirname, filename)
            image_files += [image_file]

    random.shuffle(image_files)

    num_of_images = len(image_files)
    num_of_val_images = int(0.1 * num_of_images)

    val_image_files = image_files[0:num_of_val_images]
    train_image_files = image_files[num_of_val_images:]

    for image_file in val_image_files:
        print(image_file)
        image_class = int(image_file.split("/")[-2])
        copy_dst_dir = "splitdata/validation/%s" % (str(image_class).zfill(5))
        shutil.copy(image_file, copy_dst_dir)

    train_image_files.sort()
    last_stream_num = ""
    last_site = 0
    for image_file in train_image_files:
        image_class_str, image_filename = image_file.split("/")[-2:]
        image_class = int(image_class_str)
        stream_num = image_filename.split('_')[0]

        if stream_num == last_stream_num:
            site = last_site
        else:
            if image_class % 4 == 0:
                site = np.random.choice(4, 1, p=[0.7, 0.1, 0.1, 0.1])[0]
            elif image_class % 4 == 1:
                site = np.random.choice(4, 1, p=[0.1, 0.7, 0.1, 0.1])[0]
            elif image_class % 4 == 2:
                site = np.random.choice(4, 1, p=[0.1, 0.1, 0.7, 0.1])[0]
            else:
                site = np.random.choice(4, 1, p=[0.1, 0.1, 0.1, 0.7])[0]

        copy_dst_dir = "splitdata/train/site%d/%s" % (
            site, str(image_class).zfill(5))

        shutil.copy(image_file, copy_dst_dir)

        last_stream_num = stream_num
        last_site = site

    return


def create_dir_tree():
    num_of_site = 4
    num_of_class = 43

    Path("splitdata/validation").mkdir(parents=True, exist_ok=True)
    for image_class in range(num_of_class):
        Path("splitdata/validation/%s" %
             (str(image_class).zfill(5))).mkdir(parents=True, exist_ok=True)

    for site in range(num_of_site):
        for image_class in range(num_of_class):
            Path("splitdata/train/site%d/%s" %
                 (site, str(image_class).zfill(5))).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_dir_tree()
    main()
