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
import torch

def image_process_api(image):
    size = image.size
    ratio = float(512) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    image = image.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (512, 512))
    new_im.paste(image, ((512-new_image_size[0])//2, (512-new_image_size[1])//2))
    new_image = new_im
    numpy_data = np.asarray(new_image)
    #print(np.shape(numpy_data))
    numpy_data = numpy_data.reshape((1, 3, 512, 512))
    #print(np.shape(numpy_data))
    return torch.tensor(numpy_data).float()
