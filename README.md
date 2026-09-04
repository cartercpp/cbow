# CBOW

A Continuous Bag of Words (CBOW) model implemented from scratch in C++.

The model learns word embeddings from text and predicts a missing word using the words surrounding it. The neural network, training loop, matrix operations, softmax, and gradient calculations are implemented without an ML framework.

## Demo

The demo randomly selects a 5-word sequence and masks the center word:

```text
CONTEXT
  not a [ MASK ] Minion yeah

TOP 5
  1  rich       ━━━━━━━━━━━━━━━━  97.6%
  2  you        ────────────────   1.8%
  3  Minion     ────────────────   0.2%
  4  like       ────────────────   0.1%
  5  You        ────────────────   0.1%
```

The highest-ranked prediction is displayed in green when correct and red when incorrect.

## How it works

For a sequence

```text
w1  w2  [target]  w3  w4
```

CBOW:

1. Looks up the embedding for each context word.
2. Averages the four context embeddings.
3. Multiplies the resulting vector by an output weight matrix.
4. Applies softmax to produce a probability distribution over the vocabulary.
5. Uses cross-entropy loss during training.
6. Backpropagates the error into both the output weights and word embeddings.

The model uses:

- 50-dimensional word embeddings
- 2 context words on each side
- 100 training epochs
- Learning rate of `0.01`

## Project structure

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

`training/` trains the model and writes the learned matrices to disk.

`demo/` loads the trained parameters and continuously visualizes predictions in the terminal.

## Building

A compiler with modern C++ support is required.

### Training

```bash
cd training
g++ -std=c++23 -O2 main.cpp split.cpp -o train
./train
```

### Demo

```bash
cd demo
g++ -std=c++23 -O2 main.cpp split.cpp weights.cpp -o demo
./demo
```

> The source currently contains absolute paths to the training text/output files. Change these paths to match your local filesystem before running it.

## Implementation

The project includes custom `matrix` and `math_vector` types used for the underlying linear algebra rather than relying on an external machine-learning library.

The training process implements the CBOW forward pass and backpropagation directly in C++.
