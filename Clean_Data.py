import pandas as pd
import numpy as np
import os
from PIL import Image
import PIL
import fastapi
import pydantic
import numpy as np
import uvicorn
import boto3
import transformers
import multipart
import pandas as pd
import emoji

def Clean_Products(csv_file="Products.csv"):
    products_df = pd.read_csv(csv_file, lineterminator="\n")
    products_df = products_df.drop("Unnamed: 0", axis=1)

    names = []
    for i in products_df["product_name"]:
        name = (i.split("|")[0])
        name = emoji.replace_emoji(name, "")
        names.append(name)

    descs = []
    for i in products_df["product_description"]:
        desc = emoji.replace_emoji(i, "")
        descs.append(desc)

    products_df["product_name"] = names
    products_df["product_description"] = descs

    products_df[["Main_Category", "Sub_Category", "2", "3", "4"]] = products_df.category.str.split("/", expand=True)
    products_df["Sub_Category_2"] = products_df["2"].astype(str) + products_df["3"].astype(str) + products_df["4"].astype(str)
    products_df = products_df.drop("2", axis=1)
    products_df = products_df.drop("3", axis=1)
    products_df = products_df.drop("4", axis=1)
    products_df = products_df[["id", "product_name", "Main_Category", "Sub_Category", "Sub_Category_2", "product_description", "price", "location", "url", "page_id", "create_time"]]

    products_df['Main_Category'] = pd.factorize(products_df.Main_Category)[0]

    products_df = products_df.drop("product_name", axis = 1)
    products_df = products_df.drop("Sub_Category", axis = 1)
    products_df = products_df.drop("price", axis = 1)
    products_df = products_df.drop("location", axis = 1)
    products_df = products_df.drop("url", axis = 1)
    products_df = products_df.drop("page_id", axis = 1)
    products_df = products_df.drop("create_time", axis = 1)
    products_df = products_df.drop("Sub_Category_2", axis = 1)
    products_df.rename(columns= {"id":"product_id"}, inplace=True)

    return products_df

def Clean_Images_csv(images_csv="Images.csv"):
    images_df = pd.read_csv(images_csv, lineterminator="\n")
    images_df = images_df.drop("bucket_link", axis=1)
    images_df = images_df.drop("image_ref", axis=1)
    images_df = images_df.drop("create_time\r", axis=1)
    images_df = images_df.drop("Unnamed: 0", axis=1)
    images_df.rename(columns= {"id":"image_id"}, inplace=True)
    return images_df

def Final_df():
    merged_df = pd.merge(Clean_Images_csv(), Clean_Products(), on="product_id")
    return merged_df

if __name__ == "__main__":
    print(Final_df())