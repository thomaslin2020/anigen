import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import gdown
from config import BATCH_SIZE, NUM_WORKERS, ROOT_DIR
import os

class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir, file_name='dataset.pt', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Accessing dataset as tensor is much faster than using the filesystem
        if not root_dir:
            self.images = torch.load(f'{file_name}')
        else:
            self.images = torch.load(f'{root_dir}/{file_name}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

    def set_transform(self, transform):
        self.transform = transform


class AnimeFaceDataModule(pl.LightningDataModule):
    def __init__(self, resolution=64, data_dir: str = ROOT_DIR, batch_size: int = BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean, self.std = (
            0.7007, 0.6006, 0.5895), (0.2938, 0.2973, 0.2702)
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.Normalize(self.mean, self.std)
        ])

        if os.path.isdir(data_dir):
            self.data_dir = data_dir
        else:
            link = 'https://drive.google.com/uc?id=1vcsXeKRT77CtoI-DAaTvUEsjfEPRRIuY'
            gdown.download(link, 'dataset.pt', False)
            self.data_dir = ''
        self.data =  AnimeFaceDataset(self.data_dir, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, num_workers=self.num_workers)

    def set_transform(self, transform):
        self.data.set_transform(transform)

    def get_data_mean_std(self):
        return self.mean, self.std
