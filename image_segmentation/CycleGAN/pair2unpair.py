#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by weihang huang on 17-12-10
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from glob import glob

import PIL
import matplotlib.pyplot as plt

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
root = os.path.dirname(root)

image_dir = "maps"
mode = "test"
# mode = "test"
mode_dir = "train" if mode == "train" else "val"
base_dir = os.path.join(root, "datasets/image", image_dir, mode_dir)
target_dir = os.path.join(root, "datasets/unpair_image", image_dir)

target_dirA = os.path.join(target_dir, mode + "A")
target_dirB = os.path.join(target_dir, mode + "B")

if not os.path.exists(target_dir):
    os.mkdir(target_dir)
if not os.path.exists(target_dirA):
    os.mkdir(target_dirA)
if not os.path.exists(target_dirB):
    os.mkdir(target_dirB)

base_files = glob(base_dir + "/*.jpg")
print("total_len:",len(base_files))
for index,img_file in enumerate(base_files):
    print(index,"--",img_file)
    img = plt.imread(img_file)
    img_width = int(img.shape[1] / 2)
    imgA = img[:, :img_width, :]
    imgB = img[:, img_width:, :]
    img_basename = os.path.basename(img_file)
    output_pathA = os.path.join(target_dirA, img_basename)
    output_pathB = os.path.join(target_dirB, img_basename)
    imA = PIL.Image.fromarray(imgA)
    imB = PIL.Image.fromarray(imgB)
    imA.save(output_pathA)
    imB.save(output_pathB)
