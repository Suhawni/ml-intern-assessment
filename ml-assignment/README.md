## Task 2 (Done)

📚 Trigram Language Model — ML Intern Assignment

This project implements a Trigram (N=3) Language Model from scratch.
It includes text cleaning, rare-word handling (<UNK>), trigram probability estimation, and probabilistic text generation.

The project structure:

```
ml-assignment/
│
├── data/
│   ├── example_corpus.txt
│   └── large_corpus.txt        (optional – Gutenberg book)
│
├── src/
│   ├── ngram_model.py
│   ├── generate.py
│   ├── utils.py
│   └── clean_corpus.py
│
├── tests/
│   ├── test_ngram.py
│   └── conftest.py
│
└── evaluation.md
```

How to Run the Project:

1. Train & Generate Text (using the example corpus)

From project root:

```
python -m src.generate
```

2. Running Tests

To verify the model:

```
pytest -q
```

You should see:

```
3 passed
```

## Task 2 — Scaled Dot-Product Attention (Also Done)

How to Run Task 2

From repo root:

```
cd task2
python demo.py
```

