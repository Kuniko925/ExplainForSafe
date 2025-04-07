import numpy as np
import pandas as pd

def get_dataset(d_type="train"):

    root = f"../../dataset/SKIN/ISIC2017/"

    if d_type == "train":
        df = pd.read_csv(f"{root}ISIC-2017_Training_Part3_GroundTruth.csv")
        df = uniform_dataframe(df)
        df["filepath"] = root + "ISIC-2017_Training_Data/" + df["image_id"] + ".jpg"
        df["segpath"] = root + "ISIC-2017_Training_Part1_GroundTruth/" + df["image_id"] + "_segmentation.png"
    elif d_type == "valid":
        df = pd.read_csv(f"{root}ISIC-2017_Validation_Part3_GroundTruth.csv")
        df = uniform_dataframe(df)
        df["filepath"] = root + "ISIC-2017_Validation_Data/" + df["image_id"] + ".jpg"
        df["segpath"] = root + "ISIC-2017_Validation_Part1_GroundTruth/" + df["image_id"] + "_segmentation.png"
    elif d_type == "test":
        df = pd.read_csv(f"{root}ISIC-2017_Test_v2_Part3_GroundTruth.csv")
        df = uniform_dataframe(df)
        df["filepath"] = root + "ISIC-2017_Test_v2_Data/" + df["image_id"] + ".jpg"
        df["segpath"] = root + "ISIC-2017_Test_v2_Part1_GroundTruth/" + df["image_id"] + "_segmentation.png"

    return df

def uniform_dataframe(df):
    conditions = [
        (df["melanoma"] == 1) & (df["seborrheic_keratosis"] == 1),
        (df["melanoma"] == 1),
        (df["seborrheic_keratosis"] == 1)
    ]
    choices = [3, 1, 2]
    df["label"] = np.select(conditions, choices, default=0)
    return df