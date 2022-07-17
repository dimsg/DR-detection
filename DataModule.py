from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torchsampler import ImbalancedDatasetSampler
from data import DDR, DDR_class
import albumentations as A


class ConcatDatasets(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, i):
        idx1 = i
        idx2 = i
        if i >= len(self.dataset2):
            idx2 = i % len(self.dataset2)
        return self.dataset1[idx1], self.dataset2[idx2]

    def get_labels(self):
        return self.dataset1.get_labels()


class ConcatDatasetsTesting(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, i):
        idx1 = i
        idx2 = i
        if i >= len(self.dataset2):
            return self.dataset1[idx1], [-1 ,-1 ,-1]
            # idx2 = i % len(self.dataset2)
        return self.dataset1[idx1], self.dataset2[idx2]


def training_aug():
    train_transform = [A.OneOf([A.RandomResizedCrop(width=512, height=512, scale=(0.5, 1.0),
                                           ratio=(1.0, 1.0), interpolation=2, p=0.5),
                                A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512,
                                                w2h_ratio=1.0, interpolation=2, p=0.5), ], p=0.8),
                       A.OneOf([A.HorizontalFlip(p=0.5),
                                A.RandomRotate90(p=0.5),
                                A.Rotate(limit=(-60, 60), interpolation=2, border_mode=0,
                                         value=(0, 0, 0), mask_value=None, p=0.5), ], p=0.8),]
    return A.Compose(train_transform, p=0.7)  # p that the augmentation pipeline will apply augmentations at all


class DDRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, gt_dir, class_dir, grades, batch_size: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.grades = grades
        self.class_dir = class_dir
        self.batch_size = batch_size
        self.train_dir = self.data_dir + "train/image"
        self.train_gt_dir = self.gt_dir + "train/label"
        self.valid_dir = self.data_dir + "valid/image"
        self.valid_gt_dir = self.gt_dir + "valid/label"
        self.test_dir = self.data_dir + "test/image"
        self.test_gt_dir = self.gt_dir + "test/label"
        self.train_dr_grade_dir = self.class_dir + "train"
        self.valid_dr_grade_dir = self.class_dir + "valid"
        self.test_dr_grade_dir = self.class_dir + "test"
        self.train_dataset = DDR(self.train_dir, self.train_gt_dir, self.grades, augmentation=training_aug())
        self.valid_dataset = DDR(self.valid_dir, self.valid_gt_dir, self.grades, augmentation=None)
        self.test_dataset = DDR(self.test_dir, self.test_gt_dir, self.grades, augmentation=None)
        self.train_class_dataset = DDR_class(self.train_dr_grade_dir, self.grades, augmentation=training_aug())
        self.valid_class_dataset = DDR_class(self.valid_dr_grade_dir, self.grades, augmentation=None)
        self.test_class_dataset = DDR_class(self.test_dr_grade_dir, self.grades, augmentation=None)
        self.train_concat = ConcatDatasets(self.train_class_dataset, self.train_dataset)
        self.valid_concat = ConcatDatasets(self.valid_class_dataset, self.valid_dataset)
        self.test_concat = ConcatDatasetsTesting(self.test_class_dataset, self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_concat, batch_size=self.batch_size,
                          sampler=ImbalancedDatasetSampler(self.train_concat))

    def val_dataloader(self):
        return DataLoader(self.valid_concat, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_concat, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(DDR(self.test_dir, self.test_gt_dir, self.grades,
                              augmentation=None), batch_size=1)

    def get_grades_train_valid(self):
        return self.train_class_dataset.grades_picked, self.train_dataset.grades_picked