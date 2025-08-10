from transformers import AutoModel, AutoFeatureExtractor, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
import torch
from torch import nn


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


#----------------preprare dataset-------------------
def preprocess(example):
    pixel_values = extractor(example['image'], return_tensors='pt').pixel_values
    example['pixel_values'] = pixel_values

    tokenized = tokenizer(
        example['latex_formula'],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
    example['input_ids'] = tokenized['input_ids']
    example['attention_mask'] = tokenized['attention_mask']

    return example

tokenized_train = dataset['train'].map(preprocess, batched=True)
tokenized_validation = dataset['validation'].map(preprocess, batched=True)

tokenized_train.set_format(type="torch", columns=['pixel_values', 'input_ids', 'attention_mask'])
tokenized_validation.set_format(type="torch", columns=['pixel_values', 'input_ids', 'attention_mask'])
#---------------------------------------------------

#--------------------build custom model-------------
class ViT_GPT(nn.Module):
    def __init__(self, extractor, vit_model, gpt_model):
        super().__init__()
        self.extractor = extractor
        self.vit_model = vit_model
        self.gpt_model = gpt_model

        nn.ModuleList([self.extractor, self.vit_model, self.gpt_model])
    
    def forward(self, input_ids, attention_mask, images, labels=None):
        pixel_values = self.extractor(images, return_tensors='pt').pixel_values

        with torch.no_grad():
            encoder_hidden_states = self.vit_model(pixel_values).last_hidden_state
        
        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], device=input_ids.device).long()

        outputs = self.gpt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels
        )

        return outputs
#---------------------------------------------------

model = ViT_GPT(extractor, vit_model, lang_model)

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    images = [item["image"] for item in batch]  # list of PIL images or tensors
    labels = torch.stack([item["input_ids"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "labels": labels
    }

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    data_collator=collate_fn
)

trainer.train()