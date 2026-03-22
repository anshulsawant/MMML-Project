import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def build_math_vocab():
    # Load all mathematical answers from the training distribution
    with open("data/ground_truths.json", "r") as f:
        data = json.load(f)
    answers = list(data.values())

    # Initialize a clean Byte-Pair Encoding instance
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train a highly constrained micro-vocabulary (300 tokens covers all common chunks like `sqrt`, `10`, `0.5`)
    # We include PAD and sequential delimiters so a micro-decoder can generate multi-token answers iteratively
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=300)
    
    tokenizer.train_from_iterator(answers, trainer)
    
    # Save the tokenizer locally
    tokenizer.save("data/math_vocab.json")
    
    print("Successfully compiled custom geometric BPE vocabulary into `data/math_vocab.json`!")
    print(f"Final Vocabulary Size: {tokenizer.get_vocab_size()}")
    
    # Prove the tokenizer algorithm works on complex math notation
    test_strings = ["40", "67.5", "2sqrt(5)/5", "12/13"]
    print("\n--- Tokenization Benchmarks ---")
    for s in test_strings:
        enc = tokenizer.encode(s)
        print(f"Text: {s:<12} | Tokens: {enc.tokens} | IDs: {enc.ids}")

if __name__ == "__main__":
    build_math_vocab()
