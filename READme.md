# IMDb Sentiment Classification with PyTorch

## Overview

This project focuses on building a binary sentiment classification system using PyTorch and the IMDb movie review dataset.

The objective was to classify movie reviews as either **positive** or **negative** by exploring text preprocessing, word embeddings, and neural network architectures.

Rather than using pretrained language models, all models were built and trained from scratch.

## Dataset

- **Dataset:** IMDb Movie Reviews
- **Source:** Stanford NLP — IMDb Dataset
- **Training samples:** 25,000
- **Testing samples:** 25,000
- **Classes:** Positive (1), Negative (0)
- **Class distribution:** Balanced, with 12,500 reviews per class.

The training data was further divided into:

- Training set: 20,000 reviews
- Validation set: 5,000 reviews

The test set was reserved for final model evaluation.

## Text Preprocessing

The following preprocessing steps were implemented:

1. Removed HTML line-break tags.
2. Converted text to lowercase.
3. Tokenized reviews using whitespace tokenization.
4. Built a vocabulary using the training dataset only.
5. Replaced unknown words with `<UNK>`.
6. Converted words into numerical IDs.
7. Padded or truncated reviews to a fixed length of 256 tokens.

### Vocabulary

- Vocabulary size: 20,000
- `<PAD>`: 0
- `<UNK>`: 1

No stopword removal was performed because words such as "not" can significantly change sentiment.

## Models Implemented

Three neural network architectures were developed and compared.

### 1. Embedding + Mean Pooling

The baseline model uses:

- Embedding layer (100 dimensions)
- Mean pooling
- Linear layer (100 → 64)
- ReLU activation
- Dropout (0.3)
- Output layer (64 → 2)

The embedding vectors for each review are averaged into a single representation before classification.

Padding tokens are excluded from the mean calculation.

### 2. Original LSTM

The second model introduces sequential processing using an LSTM.

Architecture:

- Embedding layer (100 dimensions)
- LSTM (hidden size: 128)
- Dropout (0.3)
- Linear output layer (128 → 2)

`pack_padded_sequence` was used to prevent the LSTM from processing unnecessary padding tokens.

### 3. Improved LSTM

The third model introduces additional regularization to address overfitting observed in the original LSTM.

Changes:

- Added embedding dropout (0.3).
- Increased classifier dropout from 0.3 to 0.5.

The remaining architecture was kept the same.

## Training Configuration

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Framework               | PyTorch          |
| Optimizer               | Adam             |
| Learning rate           | 0.001            |
| Loss function           | CrossEntropyLoss |
| Batch size              | 64               |
| Epochs                  | 10               |
| Embedding dimension     | 100              |
| Maximum sequence length | 256              |
| Vocabulary size         | 20,000           |

Model checkpoints were saved whenever validation loss improved.

The best checkpoint was selected based on **validation loss**, not validation accuracy.

## Results

### Model Performance

| Model                    | Validation Accuracy | Test Accuracy |  Test Loss |
| ------------------------ | ------------------: | ------------: | ---------: |
| Embedding + Mean Pooling |              86.28% |    **84.88%** | **0.3618** |
| Original LSTM            |              79.80% |        79.77% |     0.4544 |
| Improved LSTM            |              84.46% |        84.22% |     0.4318 |

### Final Confusion Matrix Results

**Embedding + Mean Pooling**

```text
[[11155  1345]
 [ 2436 10064]]
```

**Improved LSTM**

```text
[[10284  2216]
 [ 1729 10771]]
```

### Classification Report — Improved LSTM

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| Negative |      0.86 |   0.82 |     0.84 |
| Positive |      0.83 |   0.86 |     0.85 |

The improved LSTM performed better at identifying positive reviews, while the mean-pooling model had stronger negative-review recall.

## Key Findings

- The mean-pooling model achieved the highest test accuracy at 84.88%.
- The original LSTM showed significant overfitting.
- Adding dropout improved the LSTM's generalization performance.
- The improved LSTM achieved 84.22% test accuracy, recovering much of the performance lost by the original LSTM.
- A more complex architecture does not automatically produce better results.

Interestingly, the simpler mean-pooling architecture outperformed both LSTM experiments on this dataset.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Hugging Face Datasets
- Jupyter Notebook

## What I Learned

Through this project, I gained practical experience with:

- Text preprocessing and tokenization.
- Vocabulary construction and word-to-index mapping.
- Word embeddings and padding.
- Building neural networks with PyTorch.
- Understanding mean pooling and sequential processing.
- Implementing LSTM architectures.
- Training loops, backpropagation, and optimization.
- Overfitting and regularization.
- Model checkpointing.
- Evaluating classification models using accuracy, precision, recall, F1-score, and confusion matrices.

## Limitations

- Basic whitespace tokenization was used.
- Punctuation was not separately processed.
- The vocabulary was limited to 20,000 tokens.
- Reviews were truncated or padded to 256 tokens.
- No pretrained embeddings or language models were used.

## Conclusion

This project explored two different approaches to text classification: mean-pooled word embeddings and sequential processing with LSTMs.

Although the LSTM architecture was more sophisticated, the embedding + mean-pooling model achieved the best overall test performance.

The main takeaway was that model complexity alone does not determine performance. Architecture selection, preprocessing, regularization, and evaluation all matter.
