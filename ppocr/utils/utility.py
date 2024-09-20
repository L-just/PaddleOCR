# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import cv2
import random
import numpy as np
import paddle
import importlib.util
import sys
import subprocess


def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


def get_check_global_params(mode):
    check_params = [
        "use_gpu",
        "max_text_length",
        "image_shape",
        "image_shape",
        "character_type",
        "loss_type",
    ]
    if mode == "train_eval":
        check_params = check_params + [
            "train_batch_size_per_card",
            "test_batch_size_per_card",
        ]
    elif mode == "test":
        check_params = check_params + ["test_batch_size_per_card"]
    return check_params

# 可以处理的格式检验
def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file, infer_list=None):
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        # 文件记录的图片地址，第一列是图片路径，第二列是标签
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
        if os.path.isfile(img_file) and _check_image_file(img_file):
            # 单张图片
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            # 目录下所有图片
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)

    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def binarize_img(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
        # use cv2 threshold binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def check_and_read(img_path):
    if os.path.basename(img_path)[-3:].lower() == "gif":
        # gif 图片
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logger = logging.getLogger("ppocr")
            logger.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            # gray image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == "pdf":
        from paddle.utils import try_import
        # 导入模块
        fitz = try_import("fitz")
        from PIL import Image

        imgs = []
        # 使用PyMuPDF库读取pdf文件
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.page_count):
                # 每一页转换为图片
                page = pdf[pg]
                """
                fitz.Matrix()可以接受两个或四个参数来构造不同的变换矩阵：
                如果提供两个参数a和d，则创建一个缩放矩阵。
                如果提供四个参数a, b, c, d，则创建一个通用的2D仿射变换矩阵。
                参数说明
                a (float): 水平方向的缩放系数。
                b (float): 垂直方向的倾斜系数。
                c (float): 水平方向的倾斜系数。
                d (float): 垂直方向的缩放系数。
                e (float): 水平方向的平移量。
                f (float): 垂直方向的平移量。
                """
                mat = fitz.Matrix(2, 2)
                # 缩放因子为2，即图片大小为原来的1/2
                # alpha=False表示不包含透明通道
                """
                page：这是指PDF中的一页，通常是从Document对象中通过索引得到的页面对象。
                matrix=None：一个可选参数，用于指定渲染时使用的矩阵。默认情况下，使用的是单位矩阵（即原始大小）。你可以通过设置不同的矩阵来缩放页面。
                clip=None：一个可选参数，定义了要渲染的区域。如果提供了这个参数，只有该矩形内的内容会被渲染到Pixmap中。
                alpha=False：一个布尔值，表示是否包含透明通道。如果设为True，则生成的Pixmap将带有Alpha通道。
                colorspace=None：指定输出的颜色空间。如果不提供，默认会根据页面内容选择颜色空间。
                alpha cs=None：当alpha=True时，指定透明通道使用的颜色空间。
                bgcolor=None：背景颜色，默认情况下，使用的是页面的背景色。如果设置了此参数，则会用提供的颜色填充背景。
                """
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                # 限制图片大小为2000像素以内
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False


def load_vqa_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    old_lines = [line.strip() for line in lines]
    lines = ["O"]
    for line in old_lines:
        # "O" has already been in lines
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    labels = ["O"]
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def check_install(module_name, install_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"Warnning! The {module_name} module is NOT installed")
        print(
            f"Try install {module_name} module automatically. You can also try to install manually by pip install {install_name}."
        )
        python = sys.executable
        try:
            subprocess.check_call(
                [python, "-m", "pip", "install", install_name],
                stdout=subprocess.DEVNULL,
            )
            print(f"The {module_name} module is now installed")
        except subprocess.CalledProcessError as exc:
            raise Exception(f"Install {module_name} failed, please install manually")
    else:
        print(f"{module_name} has been installed.")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
