# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model (Task 1) and the optional Scaled Dot-Product Attention module (Task 2).

## 📦 Installation

Make sure you have Python 3.8+ installed, then install dependencies:

```
pip install -r requirements.txt
```

If you created a virtual environment:

```
source venv/bin/activate
# or on Windows:
venv\Scripts\activate
```

## ▶️ How to Run the Model

### 1. Train and Generate Text

Run the main script:

```
python src/generate.py
```

This will:
* Read the example corpus from `data/example_corpus.txt`
* Train the trigram model
* Print generated text to the console

## 🧪 Running Tests

To run all unit tests:

```
pytest -q
```

This validates:
* Trigram training
* Text generation
* Empty/short text handling

If pytest cannot find the `src` folder, set PYTHONPATH:

```
export PYTHONPATH="."
pytest -q
```

(Usually not required.)

## 📘 Project Structure

```
ml-assignment/
│
├── src/
│   ├── ngram_model.py        # Trigram model implementation
│   ├── generate.py           # Script to run training + text generation
│   ├── utils.py              # Optional helper utilities
│   └── __init__.py
│
├── tests/
│   ├── test_ngram.py         # Unit tests for trigram model
│   └── __pycache__/
│
├── data/
│   └── example_corpus.txt    # Sample training text
│
├── attention/ (Task 2 - optional)
│   ├── scaled_attention.py   # Numpy-only scaled dot-product attention
│   └── demo.py               # Demonstration script
│
├── evaluation.md             # Explanation of design choices
├── README.md
└── requirements.txt
```

## 🧠 Evaluation Document

All design decisions—including:
* Text cleaning
* Padding
* N-gram storage
* Probability sampling
* Efficiency improvements
* Task 2 details

are described in evaluation.md.

## ✔️ Optional Task 2: Scaled Dot-Product Attention

To run the Task-2 demo:

```
python attention/demo.py
```

This uses the NumPy-only implementation in:

```
attention/scaled_attention.py
```