# Evaluation of Trigram Language Model Design

This document summarizes the design choices, implementation strategy, and reasoning behind the Trigram Language Model created as part of the AI/ML Intern Assignment.

## 1. N-Gram Count Storage Design

I used the following data structure to store trigram counts:

```python
self.trigram_counts = defaultdict(Counter)
```

Each key is a context pair `(w1, w2)` and the value is a Counter of all possible next words `w3` with their frequencies:

```python
trigram_counts[(w1, w2)][w3] = count
```

**Reasons for choosing this structure:**

- `defaultdict(Counter)` makes counting simple, clean, and memory-efficient.
- Tuple keys provide fast hashing and constant-time lookup.
- A flat dictionary is significantly faster than deeply nested dictionaries.
- `Counter` makes probability sampling easy (using frequency lists).

This design optimizes speed, readability, and scalability.

## 2. Text Cleaning & Tokenization

**Steps used:**

- Convert text to lowercase
- Remove punctuation and special characters using regex
- Split on whitespace
- Remove empty entries

I chose this pipeline because it ensures:

- Stable model behavior across all inputs
- No dependency on external NLP libraries
- Consistent vocabulary normalization

Regex cleaning `([^a-z0-9\s])` is fast enough for this assignment and handles all typical text cases.

## 3. Padding Strategy

To model sentence beginnings and endings properly, I added:

```
<s>, <s>   (two start tokens)
</s>       (one end token)
```

This helps the model generate coherent phrases by learning valid starting contexts.

**Padding implementation:**

```python
tokens = ["<s>", "<s>"] + tokens + ["</s>"]
```

**Why two `<s>` tokens?**  
Trigrams require two previous words; using two `<s>` ensures the first actual token has a valid context.

## 4. Unknown Word Handling

If the model encounters a context it has never seen before, it defaults to:

```python
return "</s>"
```

This is better than:

- producing a random invalid word
- raising errors
- stalling generation

It allows generation to stop cleanly when the model lacks enough data.

## 5. Probability Sampling & Generation

### Sampling

I convert raw counts into probabilities and use:

```python
random.choices(words, weights)
```

This ensures true probabilistic language generation, not greedy selection.

**The process:**

1. Retrieve all possible next words for a context.
2. Convert frequencies to probabilities.
3. Sample a next word based on probability.

This produces natural-sounding randomness and avoids repetitive outputs.

## 6. Generation Logic

The generation process:

1. Start with:
   ```python
   w1 = "<s>", w2 = "<s>"
   ```

2. Sample the next word based on `(w1, w2)`

3. Shift window:
   ```python
   w1 = w2, w2 = w3
   ```

4. Stop when
   - `w3 == "</s>"`
   - or `length >= max_length`

This mirrors real trigram language modeling behavior.

## 7. Efficiency Improvements

Several optimizations were applied:

- Using `zip(tokens, tokens[1:], tokens[2:])` for fast trigram iteration
- Flat `defaultdict(Counter)` instead of nested dicts
- Avoid repeated list construction inside loops
- Early exit for extremely short training texts
- Regex used only once per `fit()` call
- Probability sampling done only during generation

This results in very fast training (milliseconds) and instant generation.

## 8. Overall Design Philosophy

My design prioritizes:

- **Simplicity** → easy to read and extend
- **Efficiency** → fast enough for large corpora
- **Clean API** → `fit()` and `generate()` work intuitively
- **Robustness** → no crashing on edge cases
- **Probabilistic correctness** → proper distribution sampling

The final code is production-ready, clean, and passes all provided tests.


# Task 2: Scaled Dot-Product Attention (NumPy)

I implemented the Scaled Dot-Product Attention mechanism exactly as described in the Attention Is All You Need paper, using only NumPy. The goal was to convert the mathematical formula into clean and understandable code.

## Formula

For Query Q, Key K, and Value V:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

## Steps Implemented

### 1. Score calculation (QKᵀ)
Uses matrix multiplication to compute compatibility between queries and keys.

### 2. Scaling
Values are divided by √dₖ to prevent large dot products and gradient explosions.

### 3. Masking (optional)
Mask adds −1e9 to positions where attention should be blocked (like padding or future tokens).

### 4. Softmax
Numerically stable softmax applied along the last axis.

### 5. Weighted sum
Attention weights multiplied with V to produce the final attended representation.

