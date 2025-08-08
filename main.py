from transformers import AutoModel, AutoFeatureExtractor
from datasets import load_from_disk
import torch

model = AutoModel.from_pretrained('facebook/dino-vits16')
extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vits16')

dataset = load_from_disk('data/preprocessed_data')

for param in model.parameters():
    param.requires_grad = False