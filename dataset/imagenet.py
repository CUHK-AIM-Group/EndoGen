import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = False

        self.aug_feature_dir = None
        self.aug_label_dir = None

        n_data = len(os.listdir(feature_dir))
        self.feature_files = [f"{i}.npy" for i in range(n_data)]
        self.label_files = [f"{i}.npy" for i in range(n_data)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_dir = self.feature_dir
        label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)