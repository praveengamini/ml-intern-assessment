import random
import re
from collections import defaultdict, Counter


class TrigramModel:
    def __init__(self):
        # trigram_counts[(w1, w2)] = Counter({w3: count})
        self.trigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.trained = False

    # ---------------------
    # Internal: Fast cleaning + tokenizing
    # ---------------------
    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    # ---------------------
    # Training
    # ---------------------
    def fit(self, text):
        if not text.strip():
            self.trained = False
            return

        tokens = self._tokenize(text)

        # If too short, handle gracefully
        if len(tokens) < 2:
            self.vocab = set(tokens)
            self.trained = True
            return

        # Padding for start + end of sentence
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]

        # Build vocab
        self.vocab = set(tokens)

        # Count trigrams
        for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
            self.trigram_counts[(w1, w2)][w3] += 1

        self.trained = True

    # ---------------------
    # Internal: Efficient next-word sampling
    # ---------------------
    def _sample(self, context):
        counter = self.trigram_counts.get(context)
        if not counter:
            return "</s>"

        words, counts = zip(*counter.items())
        total = sum(counts)
        weights = [c / total for c in counts]

        return random.choices(words, weights)[0]

    # ---------------------
    # Generation
    # ---------------------
    def generate(self, max_length=50):
        if not self.trained:
            return ""

        # Very small vocab case (short text training)
        if len(self.vocab) <= 3:
            return " ".join(w for w in self.vocab if w not in ("<s>", "</s>"))

        w1, w2 = "<s>", "<s>"
        generated = []

        for _ in range(max_length):
            w3 = self._sample((w1, w2))
            if w3 == "</s>":
                break
            generated.append(w3)
            w1, w2 = w2, w3

        return " ".join(generated)
