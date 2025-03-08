import torch
from torch.utils.data import Dataset
from PIL import Image

class TransDataset(Dataset):
    def __init__(self, dataframe, img_size, transform=None):
        self.dataframe = dataframe
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["filepath"]
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)