from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
from transformers import BertModel

def text_processor(text, max_length, unbatched=True):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    tokens = tokenizer.batch_encode_plus([text], max_length=max_length, padding="max_length", truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in tokens.items()}
    with torch.no_grad():
            desc = model(**encoded).last_hidden_state.swapaxes(1,2)
    desc = desc.squeeze(0)
    if unbatched == False:
        return desc[None, :]
    else:
        return desc

if __name__ == '__main__':
    encode = text_processor("Hello, my name is James", 16, unbatched=False)
    print(encode)
    print(encode.size())