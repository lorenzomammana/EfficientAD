from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Optional
import os
from torchvision import transforms
import torch


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, _ = super().__getitem__(index)
        return sample


class ImageFolderWithPath(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.good_index = self.classes.index("good")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)

        if target == self.good_index:
            target = 0
        else:
            target = 1

        return sample, target, path


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class EfficientAdDataModule(LightningDataModule):
    def __init__(
        self,
        anomaly_data_path: str,
        imagenet_data_path: str,
        category: Optional[str] = None,
        image_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.anomaly_data_path = anomaly_data_path
        self.imagenet_data_path = imagenet_data_path
        self.category = category if category is not None else ""
        self.seed = seed

        self.default_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.transform_ae = transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2),
                transforms.ColorJitter(saturation=0.2),
            ]
        )

        self.penalty_transform = transforms.Compose(
            [
                transforms.Resize((2 * image_size, 2 * image_size)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_transform(self, image):
        return self.default_transform(image), self.default_transform(self.transform_ae(image))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            anomaly_train_dataset = ImageFolderWithoutTarget(
                os.path.join(self.anomaly_data_path, self.category, "train"),
                transform=transforms.Lambda(self.train_transform),
            )

            if os.path.exists(os.path.join(self.anomaly_data_path, self.category, "validation")):
                self.validation_dataset = ImageFolderWithoutTarget(
                    os.path.join(self.anomaly_data_path, self.category, "train"),
                    transform=transforms.Lambda(self.train_transform),
                )
            else:
                # mvtec dataset paper recommend 10% validation set
                train_size = int(0.9 * len(anomaly_train_dataset))
                validation_size = len(anomaly_train_dataset) - train_size
                rng = torch.Generator().manual_seed(self.seed)
                anomaly_train_dataset, self.validation_dataset = torch.utils.data.random_split(
                    anomaly_train_dataset, [train_size, validation_size], rng
                )

            imagenet_train_dataset = ImageFolderWithoutTarget(
                os.path.join(self.imagenet_data_path), transform=self.penalty_transform
            )

            self.train_dataset = ConcatDataset(anomaly_train_dataset, imagenet_train_dataset)

            self.test_dataset = ImageFolderWithPath(
                os.path.join(self.anomaly_data_path, self.category, "test"), transform=self.default_transform
            )
        else:
            self.test_dataset = ImageFolderWithPath(
                os.path.join(self.anomaly_data_path, self.category, "test"), transform=self.default_transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def map_normalization_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
