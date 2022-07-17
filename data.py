import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import cv2
import glob
from dataPreprocess import segm_preprocess, class_preprocess


class DDR(Dataset): # For segmentation
    def __init__(self, images_path, lesions_gt_path,
                 grades_path, augmentation=None):
        self.images_path = glob.glob(images_path + "/*.png")
        self.ex_gt_path = lesions_gt_path + "/EX"
        self.he_gt_path = lesions_gt_path + "/HE"
        self.ma_gt_path = lesions_gt_path + "/MA"
        self.se_gt_path = lesions_gt_path + "/SE"
        self.augmentation = augmentation
        self.grades_picked = [0,0,0,0,0]
        if grades_path is not None:
            df = pd.read_csv(grades_path,sep=",")
            self.dr_gt = dict(zip(df.image, df.grade))
        else:
            self.dr_gt = None

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = cv2.imread(image_path)
        if image is None:
            segm_preprocess(image_path) # Got corrupted during training - Needs fixing
            image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1].split(".")[0]
        dr_grade = int(self.dr_gt[image_name])
        ex_gt_img = cv2.imread(self.ex_gt_path + "/" + image_name + ".tif")
        he_gt_img = cv2.imread(self.he_gt_path + "/" + image_name + ".tif")
        se_gt_img = cv2.imread(self.se_gt_path + "/" + image_name + ".tif")
        ma_gt_img = cv2.imread(self.ma_gt_path + "/" + image_name + ".tif")
        ex_gt_img = cv2.cvtColor(ex_gt_img, cv2.COLOR_BGR2GRAY)
        he_gt_img = cv2.cvtColor(he_gt_img, cv2.COLOR_BGR2GRAY)
        se_gt_img = cv2.cvtColor(se_gt_img, cv2.COLOR_BGR2GRAY)
        ma_gt_img = cv2.cvtColor(ma_gt_img, cv2.COLOR_BGR2GRAY)
        ex_gt_img = ex_gt_img.astype('float32') / 255.0
        ma_gt_img = ma_gt_img.astype('float32') / 255.0
        he_gt_img = he_gt_img.astype('float32') / 255.0
        se_gt_img = se_gt_img.astype('float32') / 255.0
        image = image / 255
        image = (image - 0.5) * 2
        masks = [ex_gt_img, he_gt_img, ma_gt_img, se_gt_img]

        if self.augmentation is not None:
            transformed = self.augmentation(image=image, masks=masks)
            image = transformed['image']
            lesions_gt = transformed['masks']
            lesions_gt = np.stack(lesions_gt, axis=-1)
        else:
            lesions_gt = np.stack((ex_gt_img, he_gt_img, ma_gt_img, se_gt_img), axis=-1)

        image = image.transpose(2, 0, 1).astype('float16')  # to tensor
        lesions_gt = lesions_gt.transpose(2, 0, 1)  # to tensor
        self.grades_picked[dr_grade]+=1
        return image, lesions_gt, dr_grade

    def get_labels(self):
        labels_list = []
        if self.dr_gt is None: return None
        for idx in range(0, len(self.images_path)):
            labels_list.append(self.dr_gt[self.images_path[idx].split("/")[-1].split(".")[0]])
        return labels_list

    def get_grades_picked(self):
        return self.grades_picked


class DDR_class(Dataset):
    def __init__(self, images_path, grades_path, augmentation=None):
        self.images_path = glob.glob(images_path + "/*.png")
        self.augmentation = augmentation
        self.grades_picked = [0, 0, 0, 0, 0]
        if grades_path is not None:
            df = pd.read_csv(grades_path, sep=",")
            self.dr_gt = dict(zip(df.image, df.grade))
        else:
            self.dr_gt = None

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = cv2.imread(image_path)
        if image is None:  # Got corrupted during training - Needs fixing
            class_preprocess(image_path)
            image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1].split(".")[0]
        dr_grade = int(self.dr_gt[image_name])
        image = image / 255
        image = (image - 0.5) * 2

        if self.augmentation is not None:
            transformed = self.augmentation(image=image)
            image = transformed['image']

        image = image.transpose(2, 0, 1).astype('float16')  # to tensor
        self.grades_picked[dr_grade] += 1
        return image, dr_grade

    def get_labels(self):
        labels_list = []
        if self.dr_gt is None: return None
        for idx in range(0, len(self.images_path)):
            labels_list.append(self.dr_gt[self.images_path[idx].split("/")[-1].split(".")[0]])
        return labels_list

    def get_grades_picked(self):
        return self.grades_picked