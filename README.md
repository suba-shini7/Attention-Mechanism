# 🧠 Attention Mechanism from Scratch

A complete implementation of the **Scaled Dot-Product Attention Mechanism** from scratch using **Python** and **NumPy**. This project demonstrates the mathematical foundations of the attention mechanism used in Transformer architectures without relying on TensorFlow, PyTorch, or Hugging Face libraries.

---

## 📖 Project Overview

The **Attention Mechanism** enables a model to focus on the most relevant parts of an input sequence when generating an output. It is the core building block of modern Transformer-based models such as:

- GPT
- BERT
- T5
- LLaMA
- Gemini

This repository manually implements every step involved in the attention calculation to provide a deeper understanding of how Transformers work internally.

---

# 🎯 Objectives

- Understand the concept of Self-Attention.
- Implement Scaled Dot-Product Attention from scratch.
- Learn how Query, Key, and Value matrices are generated.
- Compute Attention Scores manually.
- Apply the Softmax function.
- Generate Attention Weights.
- Produce the Context Vector.
- Visualize intermediate computations.

---

# 🏗️ Working Principle

The Attention Mechanism follows the workflow below:

```
Input Embeddings
        │
        ▼
Generate Query (Q), Key (K), Value (V)
        │
        ▼
Compute Attention Scores
        │
        ▼
Scale Scores by √dk
        │
        ▼
Apply Softmax
        │
        ▼
Generate Attention Weights
        │
        ▼
Multiply Weights × Value Matrix
        │
        ▼
Output Context Vector
```

---

# 📐 Mathematical Formula

The Scaled Dot-Product Attention is computed as:


Where:

- **Q** = Query Matrix
- **K** = Key Matrix
- **V** = Value Matrix
- **dk** = Dimension of Key vectors

---

# ⚙️ Implementation Steps

## Step 1

Prepare Input Embeddings.

```
Sentence
↓

Word Embeddings
```

---

## Step 2

Generate Query, Key and Value matrices.
•  Query = What am I looking for? 
•  Key = What information do I have? 
•  Value = Actual information returned.

```
Q = X × Wq

K = X × Wk

V = X × Wv
```

---

## Step 3

Compute Attention Scores.

![Attention Scores](https://github.com/suba-shini7/Attention-Mechanism/blob/main/Images/Softmax%20Formula.png)

## Step 4

Scale the Scores.

```
Scaled Scores = Scores / √dk
```

Scaling prevents extremely large values that can make Softmax unstable.

---

## Step 5

Apply Softmax.

```
Attention Weights = Softmax(Scaled Scores)
```

The weights represent how much attention each word should receive.

---

## Step 6

Generate Context Vector.

```
Context = Attention Weights × V
```

The Context Vector becomes the final output of the attention layer.

---

# 📂 Project Structure

```
Attention-Mechanism/
│
├── README.md
├── attention.py
├── attention.ipynb
├── requirements.txt
│
├── images/
│   ├── Workflow.png
│   ├── Self-attention-output.png
│   ├── attention_scores.png
│   ├── Self-attention-output.png
│   ├── attention_weights.png
│   └── context_vector.png
│
└── outputs/
    ├── query_matrix.png
    ├── key_matrix.png
    ├── value_matrix.png
    └── final_output.png
```

---

# 📊 Intermediate Outputs

The implementation displays:

- Input Embeddings
- Query Matrix (Q)
- Key Matrix (K)
- Value Matrix (V)
- Attention Scores
- Scaled Scores
- Softmax Probabilities
- Attention Weights
- Context Vector

---

# 🖼️ Workflow Diagram

Replace this with your image.

## WorkFlow
![Screenshot 2023-07-26 223801](https://github.com/suba-shini7/Attention-Mechanism/blob/main/Images/WorkFlow.png)

---

# 📸 Sample Outputs

## Weight Matrix

![Weight Matrix](https://github.com/suba-shini7/Attention-Mechanism/blob/main/Images/Weight.png)

## Attention Scores

![Attention Scores](https://github.com/suba-shini7/Attention-Mechanism/blob/main/Images/Transformer_self_attention_score.png)

## Softmax
![Attention Scores](https://github.com/suba-shini7/Attention-Mechanism/blob/main/Images/Self-attention_softmax.png)

## Context Vector
![Attention Scores](https://github.com/suba-shini7/Attention-Mechanism/blob/main/Images/Self-attention-output.png)

---

# 💻 Technologies Used

- Python 3
- NumPy
- Jupyter Notebook

---

# ▶️ How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/Attention-Mechanism.git
```

Go to the project folder.

```bash
cd Attention-Mechanism
```

Install dependencies.

```bash
pip install numpy
```

Run the program.

```bash
python attention.py
```

Or open the notebook.

```bash
jupyter notebook attention.ipynb
```

---

# 📈 Applications

The Attention Mechanism is widely used in:

- Natural Language Processing
- Machine Translation
- Chatbots
- Text Summarization
- Question Answering
- Sentiment Analysis
- Large Language Models (LLMs)
- Vision Transformers (ViT)
- Speech Recognition

---

# 🚀 Future Improvements

- Multi-Head Attention
- Positional Encoding
- Transformer Encoder
- Transformer Decoder
- Complete Transformer from Scratch
- GPT-style Causal Attention

---

# 📚 References

1. Vaswani et al. (2017), **Attention Is All You Need**

https://arxiv.org/abs/1706.03762

2. The Illustrated Transformer

https://jalammar.github.io/illustrated-transformer/

3. Stanford CS224N

https://web.stanford.edu/class/cs224n/

4. Dive into Deep Learning

https://d2l.ai/

---

# 👩‍💻 Author

**Subashini**

Built from scratch for educational purposes to understand the internal working of the Attention Mechanism used in Transformer architectures.

---
