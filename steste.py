import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def find_folders_with_keyword(parent_folder, keyword):
    # 遍历父文件夹下的所有文件和文件夹
    folders = []
    for root, dirs, files in os.walk(parent_folder):
        for dir in dirs:
            if keyword in dir:
                folders.append(os.path.join(root, dir))
    return folders
