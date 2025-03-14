import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataset(d_type="train"):

    root = "../../dataset/SKIN/ISIC2019/"

    if d_type == "test":
        df = pd.read_csv(f"{root}ISIC_2019_Test_GroundTruth.csv")
        df = uniform_dataframe(df)
        df["filepath"] = root + "ISIC_2019_Test_Input/" + df["image"] + ".jpg"
        df["segpath"] = root + "SEG/" + df["image"] + "_segmentation.png"
        
    else:
        df = pd.read_csv(f"{root}ISIC_2019_Training_GroundTruth.csv")
        train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
        if d_type == "train":
            df = uniform_dataframe(train_df)
            df["filepath"] = root + "ISIC_2019_Training_Input/" + df["image"] + ".jpg"
            df["segpath"] = root + "SEG/" + df["image"] + "_segmentation.png"
            df = remove_duplicated(df)
        elif d_type == "valid":
            df = uniform_dataframe(valid_df)
            df["filepath"] = root + "ISIC_2019_Training_Input/" + df["image"] + ".jpg"
            df["segpath"] = root + "SEG/" + df["image"] + "_segmentation.png"

    df.rename(columns={"image": "image_id"}, inplace=True)
    return df

def uniform_dataframe(df):
    conditions = [
        (df["MEL"] == 1),
        (df["NV"] == 1),
        (df["BCC"] == 1),
        (df["AK"] == 1),
        (df["BKL"] == 1),
        (df["DF"] == 1),
        (df["VASC"] == 1),
        (df["SCC"] == 1)
    ]
    choices = [0, 1, 2, 3, 4, 5, 6, 7]
    df["label"] = np.select(conditions, choices, default=0)
    df["image"] = df["image"].apply(lambda x: x.removesuffix("_downsampled"))
    return df

def remove_duplicated(df):
    
    df_2017 = pd.read_csv("../../dataset/SKIN/ISIC2017/ISIC-2017_Training_Part3_GroundTruth.csv")
    df = df[~df["image"].isin(df_2017["image_id"])]

    df_2017 = pd.read_csv("../../dataset/SKIN/ISIC2017/ISIC-2017_Validation_Part3_GroundTruth.csv")
    df = df[~df["image"].isin(df_2017["image_id"])]

    df_2017 = pd.read_csv("../../dataset/SKIN/ISIC2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv")
    df = df[~df["image"].isin(df_2017["image_id"])]

    return df

def remove_downsampled(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith("_downsampled.jpg"):
            new_name = filename.replace("_downsampled", "")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")

def rename_downsampled():
    folder_path = "../../dataset/SKIN/ISIC2019/ISIC_2019_Training_Input/"
    remove_downsampled(folder_path)
    folder_path = "../../dataset/SKIN/ISIC2019/ISIC_2019_Test_Input/"
    remove_downsampled(folder_path)
    print("Complete.")