import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import isic2019

class DeepLabV3(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.base_model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_class,
            kernel_size=1
        )
    def forward(self, x):
        x = self.base_model(x)
        return x["out"]


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, img_size):
        self.dataframe = dataframe
        self.img_size = img_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx]["filepath"]
        image_id = self.dataframe.iloc[idx]["image_id"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        image = preprocess(image)
        return image, image_id

class ModelEvaluator:
    def evaluate(self, seg_model, test_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seg_model.to(device)
        seg_model.eval()
        
        preds = []
        image_idxs=[]
        
        with torch.no_grad():
            for images, image_id in test_loader:
                images = images.to(device)
                outputs = seg_model(images)
                outputs = torch.sigmoid(outputs).cpu().numpy()
                outputs = (outputs > 0.5).astype("uint8")
                preds.extend(outputs)
                image_idxs.extend(image_id)
        return preds, image_idxs

def segmentation(d_type="train", save_folder="../../dataset/SKIN/ISIC2019/SEG/", model_save_path = "../../dataset/SEG/HAM/models/DeepLabV3/model_19.pt"):
    img_size = (224, 224)
    batch_size = 16
    df = isic2019.get_dataset(d_type)
    test = SkinLesionDataset(df, img_size)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    num_class=1
    seg_model = DeepLabV3(num_class)
    seg_model.load_state_dict(torch.load(model_save_path))
    
    evaluator = ModelEvaluator()
    preds, image_idxs = evaluator.evaluate(seg_model, test_loader)
    print(f"len(preds): {len(preds)}")
    print(f"len(image_idxs): {len(image_idxs)}")

    for i, pred in enumerate(preds):
        save_path = f"{save_folder}{image_idxs[i]}.png"
        Image.fromarray(pred.squeeze() * 255).save(save_path)
    print("Segmentation masks saved!")