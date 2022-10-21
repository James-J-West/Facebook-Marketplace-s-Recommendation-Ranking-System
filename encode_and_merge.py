from image_processor import image_process
from Clean_Data import Final_df
from tqdm import tqdm
import numpy as np
from text_processor import text_processor

def encode_images():
    data = Final_df()

    encoded_images = []
    for id in tqdm(data["image_id"]):
        path = f"images/{id}.jpg"
        encoded_images.append(image_process(path))
    data["encoded_image_data"] = encoded_images
    return data

if __name__ == "__main__":
    print(encode_images())
    