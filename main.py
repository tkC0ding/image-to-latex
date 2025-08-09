from transformers import AutoModel, AutoFeatureExtractor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
import torch


#global variables
ViT = 'facebook/dino-vits16'
DATA_DIR = 'data/preprocessed_data'

dataset = load_from_disk(DATA_DIR)

#---------setting up the vision transformer---------
vit_model = AutoModel.from_pretrained(ViT)
extractor = AutoFeatureExtractor.from_pretrained(ViT)

for param in vit_model.parameters():
    param.requires_grad = False
#---------------------------------------------------


#---------setting up the language model-------------
lang_model = GPT2LMHeadModel.from_pretrained("gpt2")
#---------------------------------------------------