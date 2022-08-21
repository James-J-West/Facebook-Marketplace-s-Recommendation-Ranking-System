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
import os

filename = os.path.realpath(os.path.join(os.path.dirname("__file__"), 'FACEBOOK_MARKET', 'Products.csv'))
filename2 = os.path.realpath(os.path.join(os.path.dirname("__file__"), 'FACEBOOK_MARKET', 'Products_clean.csv'))

df = pd.read_csv(filename, lineterminator="\n")

df_1 = df.drop("Unnamed: 0", axis=1)
df_1 = df_1.dropna()

names = []
for i in df_1["product_name"]:
    name = (i.split("|")[0])
    name = emoji.replace_emoji(name, "")
    names.append(name)

descs = []
for i in df_1["product_description"]:
    desc = emoji.replace_emoji(i, "")
    descs.append(desc)

df_2 = df_1
df_2["product_name"] = names
df_2["product_description"] = descs

df_3 = df_2
df_3[["Main_Category", "Sub_Category", "2", "3", "4"]] = df_3.category.str.split("/", expand=True)
df_3["Sub_Category_2"] = df_3["2"].astype(str) + df_3["3"].astype(str) + df_3["4"].astype(str)
df_3 = df_3.drop("2", axis=1)
df_3 = df_3.drop("3", axis=1)
df_3 = df_3.drop("4", axis=1)
df_3 = df_3[["id", "product_name", "Main_Category", "Sub_Category", "Sub_Category_2", "product_description", "price", "location", "url", "page_id", "create_time"]]

category = []
for i in df_3["Sub_Category_2"]:
    category.append(i.split("None")[0])

df_3["Sub_Category_2"] = category

df_3 = df_3.replace(r'^\s*$', np.nan, regex=True)

prices = []
for i in df_3["price"]:
    price = i.split("£")[1]
    prices.append(price)


df_3["price"] = prices

df_3= df_3.drop("Sub_Category_2", axis=1)

df_3.to_csv(filename2, index=False)