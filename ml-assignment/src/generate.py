from src.ngram_model import TrigramModel

def main():
    model = TrigramModel()

    try:
        with open("data/example_corpus.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("data/example_corpus.txt not found. Please provide a corpus in data/ and try again.")
        return

    model.fit(text)

    generated_text = model.generate()
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
