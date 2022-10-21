import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from torchvision import models
##############################################################
# TODO                                                       #
# Import your image and text processors here
from text_processor import text_processor
from image_processor import image_process_api
from probability import calc_prob
##############################################################
import numpy as np


class TextClassifier(nn.Module):
    def __init__(self,
                 decoder: dict=None, input_size: int=768, num_classes: int=13):
        super(TextClassifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=32, stride=1, padding="same"),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(128, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, num_classes))
        pass
##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the text model    #
##############################################################
        
        self.decoder = decoder
    def forward(self, text):
        x = self.main(text)
        return x

    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            pred = self.predict(text)
            prob = calc_prob(pred[0])
            return prob

    def predict_classes(self, text):
        with torch.no_grad():
            pred = self.predict(text)
            keys = list(self.decoder.keys())
            largest_node = torch.argmax(pred[0]).item()
            cat = keys[largest_node]
            return cat

class resnet50CNN(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__() 
        self.features = models.resnet50(pretrained=True)
        self.freeze()
        input_shape = self.features.fc.in_features
        self.features.fc = (torch.nn.Sequential(
            torch.nn.Linear(input_shape, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, out_size),
            torch.nn.Softmax(dim=1)
        ))
        self.unfreeze()
    
    def forward(self, x):
        x = self.features(x)
        return x
    
    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad=False
    
    def unfreeze(self):
        for param in self.features.fc.parameters():
            param.requires_grad=True

class ImageClassifier(nn.Module):
    def __init__(self,
                 decoder: dict=None, out_size: int=13):
        super().__init__()
        self.decoder = decoder
        self.features = models.resnet50(pretrained=True)
        self.freeze()
        input_shape = self.features.fc.in_features
        self.features.fc = (torch.nn.Sequential(
            torch.nn.Linear(input_shape, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, out_size),
            torch.nn.Softmax(dim=1)
        ))
        self.unfreeze()
    
    def forward(self, x):
        x = self.features(x)
        return x
    
    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad=False
    
    def unfreeze(self):
        for param in self.features.fc.parameters():
            param.requires_grad=True

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            print(x)
            return x
    
    def predict_proba(self, image):
        with torch.no_grad():
            pred = self.predict(image)
            prob = calc_prob(pred[0])
            return prob

    def predict_classes(self, image):
        with torch.no_grad():
            pred = self.predict(image)
            keys = list(self.decoder.keys())
            largest_node = torch.argmax(pred[0]).item()
            cat = keys[largest_node]
            return cat



class CombinedModel(nn.Module):
    def __init__(self,
                 decoder: dict=None, Text_model=None, Image_model=None):
        super(CombinedModel, self).__init__()
##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the combined model#
##############################################################
        self.Text_model = Text_model
        self.Image_model = Image_model
        self.classifier = nn.Sequential(nn.Linear(26, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 13),
                                        nn.Softmax())
        self.decoder = decoder

    def forward(self, image_features, text_features):
        x1 = self.Text_model(text_features)
        x2 = self.Image_model(image_features)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            pred = self.predict(image_features, text_features)
            prob = calc_prob(pred[0])
            return prob

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            pred = self.predict(image_features, text_features)
            keys = list(self.decoder.keys())
            largest_node = torch.argmax(pred[0]).item()
            cat = keys[largest_node]
            return cat



# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# text_decoder.pkl                                           #
##############################################################
    with open("decoder.pkl", "rb") as handle:
        text_model = TextClassifier(decoder = pickle.load(handle))
    text_model.load_state_dict(torch.load("FINAL_TEXT.pt"))
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    with open("decoder.pkl", "rb") as handle:
        image_model = ImageClassifier(decoder = pickle.load(handle))
    image_model.load_state_dict(torch.load("19_state.pkl"))
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the combined model#
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# combined_decoder.pkl                                       #
##############################################################
    with open("decoder.pkl", "rb") as handle:
        combined_model = CombinedModel(decoder = pickle.load(handle), Text_model=text_model, Image_model=image_model)
    combined_model.load_state_dict(torch.load("combined_model.pt"))
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the text processor that you will use to process #
# the text that you users will send to your API.             #
# Make sure that the max_length you use is the same you used #
# when you trained the model. If you used two different      #
# lengths for the Text and the Combined model, initialize two#
# text processors, one for each model                        #
##############################################################
    unbatched = False
    max_length = 32
    text_processor("text", max_length=max_length, unbatched=unbatched)
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    pass
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(text: TextItem):
  
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the text model   #
    # text.text is the text that the user sent to your API       #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################
    unbatched = False
    max_length = 32
    text_processed = text_processor(text.text, max_length=max_length, unbatched=unbatched)
    category_pred = text_model.predict_classes(text_processed)
    probability = text_model.predict_proba(text_processed)
    #return category_pred, probability
    return JSONResponse(content={
        "Category": str(category_pred), # Return the category here
        "Probabilities": [probability] # Return a list or dict of probabilities here
        })
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    image_processed = image_process_api(pil_image)
    category_pred = image_model.predict_classes(image_processed)
    probability = image_model.predict_proba(image_processed)

    return JSONResponse(content={
        "Category": str(category_pred), # Return the category here
        "Probabilities": list(probability) # Return a list or dict of probabilities here
        })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # In this case, text is the text that the user sent to your  #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################
    unbatched = False
    max_length = 32
    image_processed = image_process_api(pil_image)
    text_processed = text_processor(text, max_length=max_length, unbatched=unbatched)
    combined_cat = combined_model.predict_classes(image_processed, text_processed)
    combined_prob = combined_model.predict_proba(image_processed, text_processed)

    return JSONResponse(content={
    "Category": str(combined_cat), # Return the category here
    "Probabilities": list(combined_prob) # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)