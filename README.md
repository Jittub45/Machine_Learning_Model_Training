# Machine Learning Model Training

This repository contains various machine learning implementations and experiments.

## Contents

- **Multi-Head Attention** (`multihead_attention.py`) - Implementation of the Multi-Head Attention mechanism from the Transformer architecture
- **LSTM Project** (`lstm_project.ipynb`) - LSTM-based model implementation
- **Regularization** (`regularization.ipynb`) - Techniques for regularization in neural networks
- **Dropout** (`dropout_notebook.ipynb`) - Dropout regularization examples
- **Sentiment Analysis** (`sentiment_analysis_simplernn.ipynb`) - Simple RNN for sentiment analysis
- **Integer Encoding** (`integer_encoding_simplernn.ipynb`) - Text encoding with SimpleRNN

## Multi-Head Attention

The Multi-Head Attention implementation is based on the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

### Usage

```bash
python multihead_attention.py
```

### Requirements

- PyTorch
- NumPy

### Features

- Scalable dot-product attention
- Multi-head mechanism for parallel attention
- Configurable number of heads and model dimensions
- Dropout for regularization

## Getting Started

1. Clone the repository
2. Install required dependencies: `pip install torch numpy`
3. Run any of the notebooks or scripts

## License

MIT License
