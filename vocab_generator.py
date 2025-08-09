from datasets import load_from_disk

DATA_DIR = "data/preprocessed_data"
VOCAB_DIR = "data/latex_corpus.txt"

dataset = load_from_disk(DATA_DIR)

with open(VOCAB_DIR, 'w') as f:
    for category in dataset:
        for item in dataset[category]:
            s = item['latex_formula'] + '\n'
            f.write(s)
            print("groot")