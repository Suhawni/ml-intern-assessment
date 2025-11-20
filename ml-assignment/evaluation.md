# Evaluation

This document summarizes the design choices made while implementing the **Trigram Language Model** and the **Scaled Dot-Product Attention** mechanism for the ML Intern Assessment.

---

# Task 1 — Trigram Language Model

## Storage of N-Gram Counts

I used three main data structures based on Python’s `collections.Counter` and `defaultdict`.  
These choices ensure efficient counting, lookup, and clean logic.

### **Trigram Counts**
```python
trigram_counts[(w1, w2)][w3] = count
```
This stores the full trigram probability structure  
P(w₃ | w₁, w₂) and enables efficient sampling.

### **Bigram Counts**
Two versions are stored for backoff:
```python
(w1,)    -> Counter(w2)
(w1, w2) -> Counter(w3)
```
This prevents failures when a trigram is unseen by falling back to bigram context.

### **Unigram Counts**
```python
unigram_counts[w] = count
```
Used as a final fallback when bigram and trigram contexts do not exist.

Together, this tri → bi → uni layered structure ensures robust generation.

---

## Text Cleaning, Padding, and Unknown Words

### **Cleaning**
The text undergoes:
- lowercase conversion  
- removal of punctuation except `'`  
- whitespace normalization  
- sentence splitting using `.`, `!`, `?`

This ensures consistent, predictable tokens.

### **Padding**
Each sentence becomes:
```
<s>, <s>, token1, token2, … , tokenN, <eos>
```
Two `<s>` tokens are required to create the first trigram.

### **Unknown Words (`<UNK>`)**
A **two-pass training approach** was used:

1. First pass: count token frequencies  
2. Replace rare words (`count <= unk_threshold`) with `<UNK>`  
3. Rebuild all n-gram counts with `<UNK>` included

This greatly reduces sparsity and improves the model’s ability to generalize.

---

## Generate Function & Probabilistic Sampling

### **Core Logic**
Generation starts with:
```
w1 = <s>
w2 = <s>
```

At each step:
```
next_word ~ P(w3 | w1, w2)
```

Stopping conditions:
- `<eos>` is generated  
- maximum length reached  

### **Backoff Strategy**
If `(w1, w2)` does not exist:
1. Try bigram: P(w | w2)  
2. Else fallback to unigram sampling  

This prevents dead-ends and ensures fluent text even for unseen contexts.

### **Probability Sampling**
```python
random.choices(tokens, weights=counts)
```
Used to ensure *true probabilistic* generation (not greedy).  
The model produces varied, natural text instead of deterministic output.

---

## Additional Design Choices

### **Simple Regex Sentence Splitter**
This avoids external dependencies and keeps the implementation portable.

### **Removing `<s>` From Output**
Generated sentences exclude `<s>` tokens and stop cleanly on `<eos>`.

### **Fallback Output**
If the model cannot form any trigram (very tiny corpus), it returns a single sampled unigram.  
This prevents empty-string outputs and ensures tests pass.

### **Configurable `unk_threshold`**
- `unk_threshold = 0` works best for tiny corpora  
- Higher values generalize better for larger corpora  

This gives flexibility depending on dataset size.

---

## **Conclusion — Task 1**

The implementation follows classical NLP practices, prioritizing:
- clarity  
- correctness  
- robust handling of sparse data  
- true probabilistic behavior  
- extensibility  

The resulting model works effectively on both small and large corpora and produces coherent, varied text.

---

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
