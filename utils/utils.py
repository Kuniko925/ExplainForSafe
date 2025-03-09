import numpy as np
import pandas as pd
import os
import PIL
from PIL import Image
import shutil

def delete_subfolders(target_folder):

    if os.path.exists(target_folder):
        for item in os.listdir(target_folder):
            item_path = os.path.join(target_folder, item)
            try:
                shutil.rmtree(item_path)  # delete directory
            except NotADirectoryError:
                os.unlink(item_path)  # delete files

        print(f"Sub folders of {target_folder} was deleted.")
    else:
        print(f"{target_folder} was not exist.")


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"{directory_name} created.")
    else:
        print(f"{directory_name} aready exist.")

def resize(input_image_path, output_image_path, width, height, save=True):
    with Image.open(input_image_path) as img:
        if img.mode == "P":
            img = img.convert("RGB")
        resized_img = img.resize((width, height))
        if save:
            resized_img.save(output_image_path)
        else:
            return resized_img