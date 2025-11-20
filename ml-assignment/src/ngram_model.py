import re
import random
from collections import Counter, defaultdict

START = "<s>"
END = "<eos>"
UNK = "<UNK>"

class TrigramModel:
    def __init__(self, unk_threshold=1):
        """
        Trigram model:
          - trigram_counts[(w1,w2)][w3] = count
          - bigram_counts[(w1,)] or [(w1,w2)] used for fallback/backoff
          - unigram_counts[w] = count
        unk_threshold: words with raw count <= unk_threshold will be replaced by <UNK>.
        """
        self.trigram_counts = defaultdict(Counter)  # (w1,w2) -> Counter(w3)
        self.bigram_counts = defaultdict(Counter)   # (w1,) or (w1,w2) -> Counter(w2) or w3
        self.unigram_counts = Counter()
        self.vocab = set()
        self.unk_threshold = unk_threshold
        self.fitted = False

    # Cleaning & tokenization
    def _split_into_sentences(self, text):
        if not text or not text.strip():
            return []
        text = text.replace("\n", " ")
        sentences = re.split(r'[.!?]+\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean_sentence(self, sentence):
        s = sentence.lower()
        s = re.sub(r"[^a-z0-9'\s]+", "", s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _tokenize(self, sentence):
        if not sentence:
            return []
        return sentence.split()

    # Fitting (training)
    def fit(self, text):
        """
        Train the trigram model on text.
        Steps:
        1) Split into sentences
        2) Clean & tokenize
        3) Two-pass approach:
           - pass1: collect raw unigram counts to detect rare words
           - replace rare words with <UNK>
           - pass2: build unigram, bigram and trigram counts with padding
        """

        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        self.fitted = False

        sentences = self._split_into_sentences(text)
        if not sentences:
            self.fitted = True
            return

        # pass1: clean + tokenize and collect raw counts
        tokenized_sentences = []
        raw_counter = Counter()
        for sent in sentences:
            clean = self._clean_sentence(sent)
            tokens = self._tokenize(clean)
            if tokens:
                tokenized_sentences.append(tokens)
                raw_counter.update(tokens)

        if not tokenized_sentences:
            self.fitted = True
            return

        rare_words = {w for w, c in raw_counter.items() if c <= self.unk_threshold}

        # pass2: build counts with padding and UNK replacement
        for tokens in tokenized_sentences:
            tokens = [w if w not in rare_words else UNK for w in tokens]
            self.unigram_counts.update(tokens)

            padded = [START, START] + tokens + [END]

            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i+1], padded[i+2]
                self.trigram_counts[(w1, w2)][w3] += 1
                self.bigram_counts[(w1,)][w2] += 1
                self.bigram_counts[(w1,w2)][w3] += 1


        self.vocab = set(self.unigram_counts.keys())
        self.vocab.add(UNK)
        self.vocab.add(START)
        self.vocab.add(END)

        if UNK not in self.unigram_counts:
            self.unigram_counts[UNK] = 0

        self.fitted = True

    # Sampling helpers
    def _sample_from_counter(self, counter):
        """
        Given a Counter mapping token->count, sample one token proportionally to counts.
        If counter is empty, return None.
        """
        if not counter:
            return None
        items = list(counter.items())
        tokens, weights = zip(*items)
        total = sum(weights)
        # random.choices handles weights
        choice = random.choices(tokens, weights=weights, k=1)[0]
        return choice

    def _get_next_word(self, w1, w2):
        """
        Return next word sampled from trigram distribution P(w3 | w1, w2).
        Backoff strategy:
          - if (w1,w2) present, sample from trigram_counts[(w1,w2)]
          - elif (w2,) present in bigram_counts, sample from bigram_counts[(w2,)] (i.e., P(next | w2))
          - else sample from unigram_counts
        """
        trigram_counter = self.trigram_counts.get((w1, w2))
        if trigram_counter and len(trigram_counter) > 0:
            return self._sample_from_counter(trigram_counter)

        bigram_counter = self.bigram_counts.get((w2,))
        if bigram_counter and len(bigram_counter) > 0:
            return self._sample_from_counter(bigram_counter)

        if self.unigram_counts:
            return self._sample_from_counter(self.unigram_counts)

        return None

    # Generation
    def generate(self, max_length=50):
        """
        Generate text from the trained trigram model.
        Returns empty string if model was not trained on anything / empty corpus.
        """
        if not self.fitted:
            return ""

        if sum(self.unigram_counts.values()) == 0:
            return ""

        w1, w2 = START, START
        output = []

        for _ in range(max_length):
            next_word = self._get_next_word(w1, w2)
            if next_word is None:
                break
            if next_word == END:
                break
            if next_word != START:
                output.append(next_word)
            w1, w2 = w2, next_word

        if not output:
            candidate = None
            attempts = 0
            while attempts < 10:
                candidate = self._sample_from_counter(self.unigram_counts)
                if candidate and candidate not in (START, END):
                    break
                attempts += 1
            if candidate:
                return candidate
            return ""

        return " ".join(output)
