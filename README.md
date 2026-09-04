# CBOW from Scratch in C++

A from-scratch implementation of **Continuous Bag of Words (CBOW)** in modern C++.

The model learns word embeddings by predicting a target word from the words surrounding it. It uses a **context window of 4**, meaning each prediction is based on the **4 words before and 4 words after** the target word.

No machine learning libraries are used.

## How It Works

For each training example, the model takes a sequence like:

```text
word word word word [TARGET] word word word word
```

The eight surrounding words are used as context.

Each context word is mapped to a **50-dimensional embedding**. The embeddings are averaged together to produce a single context representation:

```text
8 context words
      ↓
word embeddings
      ↓
average embeddings
      ↓
linear output layer
      ↓
softmax
      ↓
predicted target word
```

The model is trained using:

* 50-dimensional word embeddings
* a context window of 4
* softmax output probabilities
* cross-entropy loss
* backpropagation
* gradient descent

The implementation includes the forward pass, loss calculation, gradient computation, parameter updates, and embedding generation directly in C++.

## Context Window

The context window is:

```cpp
constexpr std::size_t windowSize = 4;
```

For a target word at position `i`, the model uses:

```text
i - 4
i - 3
i - 2
i - 1
[TARGET]
i + 1
i + 2
i + 3
i + 4
```

This gives the model **8 context words per training example**.

The context embeddings are averaged before being passed through the output weight matrix.

## Training

The `training/` directory contains the CBOW training implementation.

Training consists of:

1. Tokenizing the input text
2. Building the vocabulary
3. Assigning each word an ID
4. Randomly initializing the embedding and output matrices
5. Creating 9-word training samples
6. Averaging the eight context embeddings
7. Computing vocabulary logits
8. Applying softmax
9. Calculating cross-entropy loss
10. Backpropagating the error
11. Updating the weights and word embeddings

The trained matrices are then saved for use by the demo.

## Demo

The `demo/` directory contains a terminal visualization of the trained model.

It randomly selects samples from the corpus, hides the center word, and asks the model to predict it from the eight surrounding words.

Example:

```text
CONTEXT

  word word word word [ MASK ] word word word word

TOP 5

  1  prediction1  ━━━━━━━━━━━━━───  72.4%
  2  prediction2  ━━━─────────────  14.1%
  3  prediction3  ━───────────────   6.3%
  4  prediction4  ────────────────   4.5%
  5  prediction5  ────────────────   2.7%
```

The demo displays the model's five most likely predictions and their softmax probabilities.

## Project Structure

```text
cbow/
├── training/
│   ├── main.cpp
│   ├── matrix.h
│   ├── math_vector.h
│   ├── split.cpp
│   ├── yeat.txt
│   ├── embeddings.txt
│   └── weights.txt
│
└── demo/
    ├── main.cpp
    ├── matrix.h
    ├── math_vector.h
    ├── split.cpp
    ├── weights.cpp
    ├── embeddings.txt
    ├── weights.txt
    └── yeat.txt
```

## Implementation

The project implements the required linear algebra without an external ML framework.

`matrix.h` and `math_vector.h` provide operations used during training, including:

* vector addition and subtraction
* scalar multiplication
* dot products
* matrix multiplication
* matrix-vector multiplication
* outer products
* matrix transposition

The CBOW training algorithm itself is implemented directly in `training/main.cpp`.

## Requirements

A modern C++ compiler with support for the C++ features used by the project is required.

The project does **not** depend on TensorFlow, PyTorch, Eigen, or another machine learning library.

## Why?

This project is meant to demonstrate how a basic neural language model works underneath high-level machine learning frameworks.

Rather than calling a training API, the implementation explicitly performs the embedding lookup, forward pass, softmax, loss calculation, backpropagation, and weight updates.
