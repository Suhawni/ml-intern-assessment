Evaluation

This document summarizes the design choices made while implementing the Trigram Language Model for the ML Intern Assessment.

1. Storage of N-Gram Counts

I used three main data structures, all based on Python’s collections.Counter and defaultdict, for fast, compact, and intuitive storage of counts:

```python
trigram_counts[(w1, w2)][w3] = count
```

This stores all observed trigrams and allows fast lookup for the probability
P(w3 | w1, w2).
Choosing a (tuple → Counter) structure keeps the logic simple, avoids nested dictionaries, and works efficiently for large corpora.

Bigram Counts

Two forms were stored to enable backoff logic:

1. (w1,) → Counter(w2)

2. (w1, w2) → Counter(w3)

This avoids failures when unseen trigram contexts appear during generation.

```python
unigram_counts[w] = count
```
Used as the final fallback for unknown contexts during text generation.

This layered structure (tri → bi → uni) closely follows classical language modeling.

2. Text Cleaning, Padding, and Unknown Words
Cleaning

The text is cleaned using:

- lowercasing

- removing punctuation except apostrophes

- reducing multiple spaces

- splitting sentences on ., !, ?

This ensures consistency in the token stream and reduces noise in the vocabulary.

Padding

Every sentence is padded with:

```
<s>, <s>   ...tokens...   <eos>
```

Two `<s>` tokens are required for trigram history.
Padding ensures the model can generate complete sentences even from small datasets.

Unknown Words (<UNK>)

I used a two-pass training approach:

1. First pass: count raw word frequencies

2. Words with count ≤ unk_threshold (default = 1) become <UNK>

3. Second pass: replace rare words with <UNK> and build n-gram counts

This prevents sparsity, handles rare words robustly, and improves generation quality, especially on small corpora.

3. Generate Function & Probabilistic Sampling
Core Logic

Generation starts with:

```
w1 = <s>, w2 = <s>
```
Then repeatedly:
```
next_word = sample P(w3 | w1, w2)
```
Stop conditions:

<eos> is generated

max_length is reached

Backoff Strategy

If (w1, w2) is unseen:

1. Try bigram: P(w | w2)

2. Else fallback to unigram sampling

This avoids dead-ends and allows coherent generation even for unseen sequences.

Sampling

I used Python’s built-in:

```python
random.choices(tokens, weights=counts)
```
4. Additional Design Choices
Simple Sentence Splitter

For ease and transparency, I used a regex-based splitter instead of external NLP libraries.
This keeps the project portable and dependency-free.

START/END Tokens Not Returned to Output

Generation excludes `<s>` tokens and stops cleanly at <eos>, producing readable sentences.

Fallback Single Token on Empty Output

For very small corpora, if no trigram can be formed, the model returns a single sampled unigram token.
This ensures tests pass and avoids empty output.

Configurable UNK Threshold

Allows:

unk_threshold = 0 for tiny corpora

higher thresholds for large corpora

making the model flexible across datasets.

Conclusion

The design choices focus on:

clarity

correctness

classical NLP practices

robustness to tiny and large datasets

true probabilistic behavior

The implementation is modular, easily extensible, and suitable for both educational and practical language modeling experiments.



## Task 2 — Scaled Dot-Product Attention (Also Done)

For Task 2, I implemented the Scaled Dot-Product Attention mechanism using only NumPy,
following the Transformer paper (“Attention Is All You Need”).

The function computes:

    scores = (Q @ Kᵀ) / sqrt(d_k)

A mask can optionally be applied by assigning -1e9 to masked positions before softmax.
Softmax is computed in a numerically stable way by subtracting the max value in each row.
The output is computed as:

    output = softmax(scores) @ V

The implementation is located in:
    task2/attention.py

A working demonstration using small Q, K, V matrices is provided in:
    task2/demo.py
