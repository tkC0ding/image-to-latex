from datasets import load_dataset, DatasetDict
from PIL import Image
import matplotlib.pyplot as plt

ds = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train")

def resize_with_padding(img, target_size=224, pad_value=255):
    img = img.convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded_img = Image.new("RGB", (target_size, target_size), (pad_value, pad_value, pad_value))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded_img.paste(img, (paste_x, paste_y))
    
    return padded_img

def transform(example):
    img = example['image']
    resized = resize_with_padding(img)
    example['image'] = resized
    return example

ds = ds.map(transform)

split_dataset = ds.train_test_split(test_size=0.2, seed=42)

val_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = DatasetDict({
    "train" : split_dataset["train"],
    "validation" : val_test_split["train"],
    "test" : val_test_split["test"]
})

final_dataset.save_to_disk("data/preprocessed_data")