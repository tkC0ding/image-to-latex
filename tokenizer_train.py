from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

corpus_path = "data/latex_corpus.txt"
save_dir = "data/latex_tokenizer"

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[corpus_path],
    vocab_size=50257,  
    min_frequency=2,   
    special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ]
)

Path(save_dir).mkdir(parents=True, exist_ok=True)
tokenizer.save_model(save_dir)

print(f"Tokenizer saved to {save_dir}")