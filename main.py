from datasets import load_dataset, DatasetDict

ds = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train")

split_dataset = ds.train_test_split(test_size=0.2, seed=42)

val_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = DatasetDict({
    "train" : split_dataset["train"],
    "validation" : val_test_split["train"],
    "test" : val_test_split["test"]
})

print(final_dataset)