import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class TransDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]["image_id"]
        filepath = self.dataframe.iloc[idx]["filepath"]
        image = Image.open(filepath).convert("RGB")
        label = self.dataframe.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), image_id

def get_dataloader(df, img_size, batch_size, eval=False):
    transform = None
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    
    if eval:
        transform = transforms.Compose([
                    transforms.RandomResizedCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                ])
    else:
        transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    
    dataset = TransDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=eval) # Train = True, Valid or Test = False
    return loader