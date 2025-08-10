from transformers import AutoModel, AutoFeatureExtractor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_from_disk
import torch


#global variables
ViT = 'facebook/dino-vits16'
DATA_DIR = 'data/preprocessed_data'
TOKENIZER = 'data/latex_tokenizer'

dataset = load_from_disk(DATA_DIR)

#---------setting up the vision transformer---------
vit_model = AutoModel.from_pretrained(ViT)
extractor = AutoFeatureExtractor.from_pretrained(ViT)

for param in vit_model.parameters():
    param.requires_grad = False
#---------------------------------------------------


#---------setting up the language model-------------
lang_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained(
    TOKENIZER,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    mask_token="<mask>"
)
lang_model.resize_token_embeddings(len(tokenizer))
#---------------------------------------------------