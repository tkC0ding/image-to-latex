from transformers import AutoModel, AutoFeatureExtractor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
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
tokenizer = GPT2TokenizerFast.from_pretrained(
    TOKENIZER,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    mask_token="<mask>"
)

config = GPT2Config.from_pretrained("gpt2", add_cross_attention=True)
lang_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
lang_model.resize_token_embeddings(len(tokenizer))
#---------------------------------------------------


#---------------Lora configuration------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

lang_model = get_peft_model(lang_model, lora_config)
lang_model.print_trainable_parameters()
#---------------------------------------------------