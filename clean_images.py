from distutils.command.clean import clean
import fastapi
import pydantic
import numpy as np
import uvicorn
import boto3
import transformers
import multipart
import pandas as pd
import emoji
from PIL import Image
import os
import numpy as np
from pathlib import Path

#RESIZE THE IMAGE
def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

#OPEN THE IMAGES AND RESIZE THEM
def clean_image_data(parent_foldername, image_folder):
    folder_path = os.path.realpath(os.path.join(os.path.dirname("__file__"), f'{parent_foldername}', 'Resized_Images'))
    os.mkdir(folder_path)
    for root, dirs, files in os.walk(os.path.realpath(os.path.join(os.path.dirname("__file__"), f'{parent_foldername}', f'{image_folder}\\'))):
        for count, file in enumerate(files):
            print(f'Resizing Image {count+1}')
            path = os.path.join(root, file)
            name = (path.split("\\")[-1].split(".jpg")[0])
            image = Image.open(path)
            new_image = resize_image(512, image)
            new_image.save(f'{folder_path}/{name}_resized.jpg')

clean_image_data("FACEBOOK_MARKET","images")